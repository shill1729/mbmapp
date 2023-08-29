# We implement a simple feed forward neural network for learning vector fields of R^D
# The class has methods for computing the Jacobian of the network with respect to its
# input vectors. Therefore, it can be used to model metric tensors as g = Df^T Df where
# f is the network, or even obtain orthogonal projection matrices via the formula
# P = I-Df^T (Df Df^T)^{-1} Df, etc.
from activations import ACTIVATION_FUNCTIONS, ACTIVATION_DERIVATIVES
import torch.nn as nn
import torch.optim


class FeedForwardNeuralNet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_layers,
                 activation="tanh",
                 final_act = "id",
                 *args,
                 **kwargs):
        """
        Learn a vector field of R^D as a feed-forward neural network

        :param extrinsic_dim: the (large) dimension of the data
        :param hidden_dim: the repeated hidden dimension used in the inner L layers of the neural nets
        :param num_layers: the number of layers in each neural net
        :param activation: the activation function, either "tanh", "elu", "sigmoid", "gelu", "softplus",
        "gaussian", "silu" or "celu"
        :param final_act: the final activation function
        :param args: args to pass to nn.Module
        :param kwargs: kwargs to pass to nn.Module
        """
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._affine_list = nn.ModuleList()
        self.mse = nn.MSELoss()
        self.act_fun = activation
        self.final_act = final_act
        if isinstance(activation, str):
            self.a = ACTIVATION_FUNCTIONS[activation]
        else:
            raise ValueError("activation must be a string naming the activation function")
        if isinstance(final_act, str):
            self.af = ACTIVATION_FUNCTIONS[final_act]
        else:
            raise ValueError("'final_act' must be a string naming the activation function")
        self._initialize_nets()

    def _initialize_nets(self):
        """ Initialize the lists containing the linear layers.
        """
        # Decoder neural net input_dim -> d1
        self._affine_list.append(nn.Linear(self.input_dim, self.hidden_dim))
        # Hidden layers
        for i in range(self.num_layers):
            self._affine_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # Final layers from d1 to output_dim
        self._affine_list.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        """
        Map x in R^D to mu(x) in R^D

        :param x: torch tensor of shape (n, D) or (1, D)
        :return: torch tensor of shape (n, d) or (1, d)
        """
        # There are num_layers+2 total layers for the map from D to d
        for i in range(self.num_layers + 1):
            x = self.a(self._affine_list[i](x))
        x = self._affine_list[-1](x)
        return self.af(x)

    def _get_weights(self):
        """
        Get the weights of each linear layer of the decoder neural net
        :return:
        """
        weights = []
        for i in range(self.num_layers + 2):
            weights.append(self._affine_list[i].weight)
        return weights

    def _compute_jacobian(self, x, weights, function_list):
        """
        Compute th e Jacobian matrix of NN(x)=FC_L o act o ... act o FC_1

        :param x: intrinsic input vector (can be batched)
        :param weights: weights of the decoder or encoder
        :param function_list: decoder or encoder list
        :return: matrix
        """
        # Wrapper for single inputs to batches
        if len(x.size()) == 1:
            x = x.view(1, x.size()[0])
        y = x
        n = x.size()[0]
        D = weights[0].repeat(n, 1, 1)
        y = function_list[0](y)
        for i in range(1, self.num_layers + 2):
            # Multiply matrices
            diag_term = torch.diag_embed(ACTIVATION_DERIVATIVES[self.act_fun](y))
            next_term = torch.bmm(weights[i].repeat(n, 1, 1), diag_term)
            D = torch.bmm(next_term, D)
            # Update input
            y = function_list[i](self.a(y))
        return D

    def jacobian_network(self, x):
        """
        Compute the Jacobian matrix of phi(u) in the manifold parameterization (phi(u))^T
        :param u: intrinsic input vector (can be batched) of size (n,d)
        :return: matrix
        """
        weights = self._get_weights()
        J = self._compute_jacobian(x, weights, self._affine_list)
        if self.final_act != "id":
            # The implicit function Jacobian has one extra matrix multiplication pertaining to
            # the final activation pass
            derivative = ACTIVATION_DERIVATIVES[self.act_fun](self.forward(x))
            a = torch.diag_embed(derivative)
            # TODO: This is required for nimpf but not auto encoder?
            # a = a.view(1, a.size()[0], a.size()[1])
            J = torch.bmm(a, J)
        return J


class ExtrinsicDriftNeuralNetwork(nn.Module):
    def __init__(self,
                 extrinsic_dim,
                 hidden_dim,
                 num_layers,
                 activation="tanh",
                 *args,
                 **kwargs):
        """
        Learn a vector field of R^D as a feed-forward neural network

        :param extrinsic_dim: the (large) dimension of the data
        :param hidden_dim: the repeated hidden dimension used in the inner L layers of the neural nets
        :param num_layers: the number of layers in each neural net
        :param activation: the activation function, either "tanh", "elu", "sigmoid", "gelu", "softplus",
        "gaussian", "silu" or "celu"
        :param args: args to pass to nn.Module
        :param kwargs: kwargs to pass to nn.Module
        """
        super().__init__(*args, **kwargs)
        self.extrinsic_dim = extrinsic_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mse = nn.MSELoss()
        self.act_fun = activation
        self.drift_network = FeedForwardNeuralNet(extrinsic_dim, extrinsic_dim, hidden_dim,
                                                  num_layers, activation, "id")

    def forward(self, x):
        """
        Map x in R^D to mu(x) in R^D

        :param x: torch tensor of shape (n, D) or (1, D)
        :return: torch tensor of shape (n, d) or (1, d)
        """
        return self.drift_network.forward(x)

    def loss(self, x, mu):
        """
        The loss function for training a NN to learn the drift vector field. It is the
        MSE of the model drift and the observed drift.

        :param x: extrinsic input of shape (n, D) or (1,D)
        :param mu: the extrinsic observed drift mu(x_i)
        :return: torch tensor
        """
        mu_out = self.forward(x)
        # MSE
        point_cloud_mse = self.mse(mu_out, mu)
        return point_cloud_mse

    def fit(self, points, drifts, lr, epochs, printfreq=1000):
        """
        Train a given NN

        :param drifts: the training data, expected to be of (n, D)
        :param lr, the learning rate
        :param epochs: the number of training epochs
        :param printfreq: print frequency of the training loss

        :return: None
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(epochs + 1):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            total_loss = self.loss(points, drifts)
            # Stepping through optimizer
            total_loss.backward()
            optimizer.step()
            if epoch % printfreq == 0:
                print('Epoch: {}: Train-Loss: {}'.format(epoch, total_loss.item()))
