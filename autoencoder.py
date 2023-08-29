# We implement a model as a Python Class object containing
# 2 neural networks:
# 1. An Encoder pi: R^D -> R^d
# 2. A Decoder phi: R^d -> R^D
#
# The point being that we can locally learn the geometry of a manifold as
# M = {phi(u): u in U}
#
# The decoder network is of the following form:
# phi(u) = FC_L o a o ... o a o FC_1 (u),
# This ensures that the output of this network is not constrained to any range
# so we can properly minimize the MSE of the point cloud and the auto-encoder.
# An optional contractive regularization term is added: it is the (mean square) Frobenius norm
# of the Jacobian of the encoder over the input points.
#
# The encoder network is of a similar form but with one extra activation function
# pi(x) =  a_F o FC_L o a o ... o a FC_1 (x),
# which ensures that the range of the local coordinates u = pi(x) are squashed into
# some range from the activation function, like (-1, 1) from tanh, etc.
#
# Here, FC_i are "fully-connected" layers, i.e. affine functions FC(q)=Wq+b, which differ
# for each network and a(.) is a smooth activation function (either tanh, elu, gelu, sigmoid, softplus,
# silu, or celu).
# TODO put this and nimpf.py into a python package
from activations import ACTIVATION_FUNCTIONS, ACTIVATION_RANGES
from ffnn import FeedForwardNeuralNet

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim
import datetime
import warnings


class AutoEncoder(nn.Module):
    def __init__(self,
                 extrinsic_dim,
                 intrinsic_dim,
                 hidden_dim,
                 num_layers,
                 activation="tanh",
                 *args,
                 **kwargs):
        """
        Learn a chart of a manifold using an auto-encoder, i.e. two neural nets to learn the mapping
        between points on a manifold and a low dimensional coordinate parameterization of it. Specifically,
        we use one neural net of L+2 layers mapping R^D -> R^d1 -> ... -> R^d1 -> R^d and another for
        the opposite direction. The encoder (from D -> d) has a final activation function applied, while
        the decoder does not.

        :param extrinsic_dim: the (large) dimension of the data
        :param intrinsic_dim: the (small) dimension of the parameterization
        :param hidden_dim: the repeated hidden dimension used in the inner L layers of the neural nets
        :param num_layers: the number of layers in each neural net
        :param activation: the activation function, either "tanh", "elu", "sigmoid", "gelu", "softplus",
        "gaussian", "silu" or "celu"
        :param args: args to pass to nn.Module
        :param kwargs: kwargs to pass to nn.Module
        """
        super().__init__(*args, **kwargs)
        self.time_of_explosion = None
        self.time_of_exit = None
        self.path_exploded = False
        self.chart_exited = False
        self.encoded_ball_radius = 0.
        self.fps = 100
        self.extrinsic_dim = extrinsic_dim
        self.intrinsic_dim = intrinsic_dim
        self.codim = extrinsic_dim - intrinsic_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mse = nn.MSELoss()
        self.act_fun = activation
        self.drift_lists = []
        self.sigma_lists = []
        self.volume_plots = []
        self.surface_plots = []
        self.encoder_net = FeedForwardNeuralNet(extrinsic_dim,
                                                intrinsic_dim,
                                                hidden_dim,
                                                num_layers,
                                                activation,
                                                activation)
        self.decoder_net = FeedForwardNeuralNet(intrinsic_dim,
                                                extrinsic_dim,
                                                hidden_dim,
                                                num_layers,
                                                activation)
        # TODO extend this to beyond Tanh
        self.grid_size = 50
        if isinstance(activation, str):
            ab = ACTIVATION_RANGES[activation]
            self.lb = ab[0]
            self.ub = ab[1]
        else:
            raise ValueError("activation must be a string naming the activation function")

    def set_mesh(self, a, b):
        """ Set the boundaries of the low-dimensional grid for plotting the vol-surface.
        """
        self.lb = a
        self.ub = b
        return None

    def set_fps(self, b):
        """
        Set the frequency of plots to save during training.
        :param b:
        :return:
        """
        self.fps = b
        return None

    def encoder(self, x):
        """
        Map x in R^D to u=pi(x) in R^d

        :param x: torch tensor of shape (n, D) or (1, D)
        :return: torch tensor of shape (n, d) or (1, d)
        """
        return self.encoder_net.forward(x)

    def decoder(self, u):
        """
        Map u in R^d to x=phi(u) in R^D

        :param u: torch tensor of shape (n, d) or (1,d)
        :return: torch tensor of shape (n, D) or (1, D)
        """
        return self.decoder_net.forward(u)

    def forward(self, x):
        """
        Forward method for the 1-chart neural net.

        :param x: data input z in R^D, can be batched i.e. size (n, D)
        :return: a tensor z' of shape (n, D)
        """
        u = self.encoder(x)
        x_out = self.decoder(u)
        return x_out

    def loss(self, x, ctr_reg=0., sparse_reg=0., orthog_reg=0., diffeo_reg=0.):
        """
        The loss function for the autoencoder (MSE of point cloud and auto-encoder output) plus optional
        'contractive' regularization

        :param x: extrinsic input of shape (n, D) or (1,D)
        :param ctr_reg: float for contractive regularization, default is zero.
        :param sparse_reg: float for sparse regularity (1-norm of encoded vectors)
        :param orthog_reg: float for orthogonal tangent vectors regularizaiton
        :param diffeo_Reg: float for diffemorphic regularization
        :return: torch tensor
        """
        u = self.encoder(x)
        x_hat = self.decoder(u)
        # MSE
        point_cloud_mse = self.mse(x, x_hat)
        # TODO consider refactoring this into separate private regularization methods
        # Contractive regularization
        if ctr_reg > 0:
            Dpi = self.jacobian_encoder(x)
            fro_norm_sq = torch.linalg.matrix_norm(Dpi, ord="fro") ** 2
            # The PML book uses the sum of the Frobenius norms over point cloud.
            # Using the mean leads to identifying their factor 'lambda' with our
            # 'ctr_reg/n' where n is the number of points in the cloud.
            contractive_error = torch.mean(fro_norm_sq)
        else:
            contractive_error = 0.
        # Sparse penalty
        if sparse_reg > 0:
            u_norm = torch.mean(torch.linalg.vector_norm(u, ord=1, dim=1))
        else:
            u_norm = 0.
        # Metric tensor diagonal regularization / orthogonal chart regularization
        if orthog_reg > 0:
            g = self.metric_tensor(u)
            diags = torch.diagonal(g, dim1=1, dim2=2)
            trace_term = torch.sum(diags ** 2, dim=1)
            gnorms = torch.linalg.matrix_norm(g, ord="fro") ** 2
            g_reg_term = torch.mean(gnorms - trace_term)
        else:
            g_reg_term = 0.
        diffeo_term = 0
        if diffeo_reg > 0.:
            Dpi = self.jacobian_encoder(x)
            Dphi = self.jacobian_decoder(u)
            A = torch.bmm(Dpi, Dphi)
            fro_norm_sq = torch.linalg.matrix_norm(A-torch.eye(self.intrinsic_dim), ord="fro") ** 2
            diffeo_term = torch.mean(fro_norm_sq)

        loss_value = point_cloud_mse + \
                     ctr_reg * contractive_error + \
                     sparse_reg * u_norm + \
                     g_reg_term * orthog_reg + \
                     diffeo_term * diffeo_reg
        return loss_value

    def fit(self, points, lr, epochs, ctr_reg=0., sparse_reg=0., orthog_reg=0., diffeo_reg=0.,
            save_plots=False, printfreq=1000):
        """
        Train the autoencoder on a point-cloud. Four optional regularization methods are available.

        :param points: the training data, expected to be of (n, D)
        :param lr, the learning rate
        :param epochs: the number of training epochs
        :param ctr_reg: regularization factor for contractive regularization
        :param sparse_reg: regularization factor for sparsity penalty
        :param orthog_reg: regularization factor for enforcing orthogonal tangent vectors
        :param diffe_reg: regularization factor for enforcing a property of diffeomorphisms
        :param save_plots: boolean for saving training plots
        :param printfreq: print frequency of the training loss

        :return: None
        """
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        for epoch in range(epochs + 1):
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            total_loss = self.loss(points, ctr_reg, sparse_reg, orthog_reg, diffeo_reg)
            # Stepping through optimizer
            total_loss.backward()
            optimizer.step()

            if epoch % printfreq == 0:
                print('Epoch: {}: Train-Loss: {}'.format(epoch, total_loss.item()))
            if epoch % self.fps == 0 and save_plots:
                self._store_volume_plot()
                self._store_surface_plot()
        # Once we are done fitting.

    def jacobian_decoder(self, y):
        """
        Compute the Jacobian matrix of phi(y) in the manifold parameterization (phi(y))^T
        :param u: intrinsic input vector (can be batched) of size (n,d)
        :return: matrix
        """

        return self.decoder_net.jacobian_network(y)

    def jacobian_encoder(self, x):
        """
        Compute the Jacobian matrix of pi(x) in the manifold parameterization pi:R^D to R^d
        :param x: extrinsic input vector (can be batched) of size (n,D)
        :return: matrix
        """
        return self.encoder_net.jacobian_network(x)

    def metric_tensor(self, y):
        """
        Compute the metric tensor of the local chart learned via the auto-encoder.

        :param y: local coordinate input
        :return: a (n,d,d) tensor representing n batches of dxd matrices.
        """
        j = self.jacobian_decoder(y)
        g = torch.bmm(j.mT, j)
        return g

    def inverse_metric_tensor(self, y):
        """
        Compute the inverse metric tensor of the local chart learned via the auto-encoder.

        :param y: local coordinate input
        :return: a (n,d,d) tensor representing n batches of dxd matrices.
        """
        g = self.metric_tensor(y)
        return torch.linalg.inv(g)

    def sqrt_inv_tensor(self, y):
        """
        Compute the square root of the inverse metric tensor of the local chart learned via the auto-encoder.

        :param y: local coordinate input
        :return: a (n,d,d) tensor representing n batches of dxd matrices.
        """
        g_inv = self.inverse_metric_tensor(y)
        return torch.linalg.cholesky(g_inv)

    def det_g(self, y):
        """
        Compute the determinant of the metric tensor
        :param y: local coordinate input
        :return: numeric
        """
        g = self.metric_tensor(y)
        return torch.linalg.det(g)

    def riemannian_norm(self, y):
        """
        Compute the Riemannian norm
        :param y: local coordinate input
        :return: numeric
        """
        g = self.metric_tensor(y)
        return torch.sqrt(torch.bmm(y.mT, torch.bmm(g, y)))

    def manifold_divergence(self, x, N):
        """
        Compute the manifold divergence of a matrix (row-wise).
        :param x: Local coordinate input
        :param N: Matrix to compute the divergence of, row-wise.
        :return: Vector
        """
        if len(N.size()) != 3:
            N = N.view(1, N.size()[0], N.size()[1])
        if len(x.size()) == 1:
            num_batches = 1
        else:
            num_batches = x.size()[0]
        # TODO put into a function in autocalculus.py
        divm = torch.zeros((num_batches, self.intrinsic_dim, 1))
        for l in range(num_batches):
            for i in range(self.intrinsic_dim):
                row_div = 0.
                for j in range(self.intrinsic_dim):
                    row_div += torch.autograd.grad(N[l, i, j], x, retain_graph=True)[0][j]
                divm[l, i, 0] = row_div
        return divm

    def extrinsic_drift_itos(self, x):
        """
        Compute the extrinsic drift vector via Ito's lemma on the chart.
        :param x: extrinsic input
        :return: vector
        """
        u = self.encoder(x)
        g = self.metric_tensor(u)
        g_inv = torch.linalg.inv(g)
        det_g = torch.linalg.det(g)
        # Compute manifold divergence for intrinsic local drift
        V = torch.sqrt(det_g) * g_inv
        # divV = batch_auto_divergence(V, u0) / torch.sqrt(det_g)
        divV = self.manifold_divergence(u, V) / torch.sqrt(det_g)
        mu = 0.5 * divV
        # Compute Jacobian of decoder
        Dphi = self.jacobian_decoder(u)
        first_order_term = Dphi @ mu
        # Now we compute the second order term: 0.5 Tr(g^{-1} Hessian(decoder))
        nb, D, d = Dphi.size()
        second_order_term = torch.zeros((nb, D, 1))
        # TODO put into autocalculus.py
        for l in range(nb):
            for i in range(D):
                hess = torch.zeros((d, d))
                for j in range(d):
                    hess[j, :] = torch.autograd.grad(Dphi[l, i, j], u, retain_graph=True)[0]
                trace_term = torch.sum(torch.diagonal(g_inv @ hess))
                second_order_term[l, i] = 0.5 * trace_term
        extrinsic_drift = first_order_term + second_order_term
        return extrinsic_drift

    # TODO: separate this into EM for BM or EM for Langevin?
    def euler_intrinsic(self, u0, potential_gradient=None):
        """
        Compute the SDE coefficients of a Brownian motion in a local chart of a Riemannian manifold
        :param u0: previous point, in the local coordinates
        :param potential_gradient: optional potential gradient for Langevin dynamics
        :return: tuple of the vector mu_{intrinsic} and matrix sqrt{g^{-1}}
        """
        # u0 = u0.view((1, u0.size()[0]))
        g = self.metric_tensor(u0)
        # TODO: The docs say that this is more stable than linalg.inv
        #  but it does not appear to be batched? or is it?
        g_inv = torch.linalg.solve(g, torch.eye(self.intrinsic_dim))
        det_g = torch.linalg.det(g)
        # Compute manifold divergence for intrinsic local drift
        V = torch.sqrt(det_g) * g_inv
        # divV = batch_auto_divergence(V, u0) / torch.sqrt(det_g)
        divV = self.manifold_divergence(u0, V) / torch.sqrt(det_g)
        mu = 0.5 * divV
        # Compute g-orthonormal frame:
        # TODO consider making this an option for the user rather than defaulting to Cholesky over SVD
        try:
            sigma = torch.linalg.cholesky(g_inv)
        except RuntimeError:
            warnings.warn("Cholesky decomp of the inv qmetric tensor failed. Using SVD.", RuntimeWarning)
            U, S, Vh = torch.linalg.svd(g_inv)
            S = torch.flip(S, (1,))
            U = torch.flip(U, (0, 2))
            sigma = torch.bmm(U, torch.diag_embed(torch.sqrt(S)))

        if potential_gradient is not None:
            nabla_U = potential_gradient(u0)
            nabla_g_U = g_inv @ nabla_U
            nabla_g_U = nabla_g_U.view(1, self.intrinsic_dim, 1)
            # print(nabla_g_U.size())
            return 2 * mu - nabla_g_U, np.sqrt(2) * sigma
        else:
            return mu, sigma

    def _exited(self, i, h, u):
        """
        Update attributes if a sample path has exited a chart.
        """
        self.chart_exited = True
        self.time_of_exit = i * h
        print("Exited chart at time " + str(self.time_of_exit))
        u[(i+1):, :] = u[i-1, :]
        return u

    # TODO: rename these?
    def brownian_motion(self, u0, tn, n, potential_gradient=None):
        """
        Simulate a Brownian motion on the local chart of a manifold learned via an auto-encoder
        until the first time the metric tensor is singular or not PD or until it exists the chart A^d, where
        A is the activation function's range.

        :param u0: initial point
        :param tn: time-horizon
        :param n: number of time steps
        :param potential_gradient: gradient for Langevin motion
        :return: tensor of shape (n, d)
        """
        u = torch.zeros((n + 1, self.intrinsic_dim))
        h = tn / n
        u[0, :] = u0
        for i in range(n):
            # TODO: Catch failure of Cholesky.
            try:
                mu, sigma = self.euler_intrinsic(u[i, :], potential_gradient)
            except RuntimeError:
                warnings.warn("Either inverting the metric tensor failed or CD/SVD of g^{-1} failed.", RuntimeWarning)
                local_singularity = u[i, :].detach().numpy()
                ambient_singularity = self.decoder(u[i, :]).detach().numpy()
                print("\nExplosion at time = " + str(i * h))
                print("Local singularity at u = " + str(local_singularity))
                print("Ambient singularity at x = " + str(ambient_singularity))
                self.path_exploded = True
                self.time_of_explosion = i * h
                print("Last drift")
                print(self.drift_lists[-1])
                print("Last diffusion")
                print(self.sigma_lists[-1])
                u[(i+1):, :] = u[i-1, :]
                return u
            mu = mu.view(self.intrinsic_dim)
            sigma = sigma.view(self.intrinsic_dim, self.intrinsic_dim)
            if not self.chart_exited:
                self.drift_lists.append(mu)
                self.sigma_lists.append(sigma)
            z = torch.randn(self.intrinsic_dim)
            u[i + 1, :] = u[i, :] + mu * h + np.sqrt(h) * sigma @ z
            # Check if we have exited the chart
            if not self.chart_exited:
                for j in range(self.intrinsic_dim):
                    if u[i, j] < self.lb or u[i, j] > self.ub:
                        print("Exiting chart at time = "+str(i*h))
                        print("Last drift")
                        print(self.drift_lists[-1])
                        print("Last diffusion")
                        print(self.sigma_lists[-1])
                        self.chart_exited = True
                        return self._exited(i, h, u)
        # If the path hasn't exploded,
        self.path_exploded = False
        self.time_of_explosion = torch.inf
        return u

    def solve_for_int_drift(self, x, mu):
        """
        Recover the intrinsic drift vector given the extrinsic drift and the chart.

        :param x: extrinsic/ambient coordinate input
        :param mu: extrinsic drift vector
        :return: intrinsic drift vector in lower dimensional space
        """
        u = self.encoder(x)
        Dphi = self.jacobian_decoder(u)
        g = torch.bmm(Dphi.mT, Dphi)
        g_inv = torch.linalg.inv(g)
        # Now we compute the second order term: 0.5 Tr(g^{-1} Hessian(decoder))
        nb, D, d = Dphi.size()
        second_order_term = torch.zeros((nb, D, 1))
        # TODO put into autocalculus.py
        for l in range(nb):
            for i in range(D):
                hess = torch.zeros((d, d))
                for j in range(d):
                    hess[j, :] = torch.autograd.grad(Dphi[l, i, j], u, retain_graph=True)[0]
                trace_term = torch.sum(torch.diagonal(g_inv @ hess))
                second_order_term[l, i] = 0.5 * trace_term
        b = second_order_term
        c = mu - b
        mu_tilde = g_inv[0] @ Dphi.mT[0] @ c[0]
        return mu_tilde

    def plot_surface(self, a, b, grid_size, ax=None):
        """
        Plot the surface produced by the neural-network chart.

        :param a: the lb of the encoder range box [a,b]^d
        :param b: the ub of hte encoder range box [a,b]^d
        :param grid_size: grid size for the mesh of the encoder range
        :param ax: plot axis object
        :return:
        """
        # Parametric representation of surface
        u, v = np.mgrid[a:b:grid_size * 1j, a:b:grid_size * 1j]
        x1 = np.zeros((grid_size, grid_size))
        x2 = np.zeros((grid_size, grid_size))
        x3 = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                x0 = np.column_stack([u[i, j], v[i, j]])
                x0 = torch.tensor(x0, dtype=torch.float32)
                xx = self.decoder(x0).detach().numpy()
                x1[i, j] = xx[0, 0]
                x2[i, j] = xx[0, 1]
                x3[i, j] = xx[0, 2]
        # set up the axes for the first plot
        # fig = plt.figure(figsize=(7, 7))
        # ax = plt.axes(projection='3d')
        if ax is not None:
            ax.plot_surface(x1, x2, x3, alpha=0.5, cmap="plasma")
            ax.set_title("NN manifold")
        else:
            return x1, x2, x3

    def plot_det_g(self, a, b, grid_size, ax=None):
        """
        Plot the volume density surface of the trained Chart-Autoencoder

        :param a: the lb of the encoder range box [a,b]^d
        :param b: the ub of hte encoder range box [a,b]^d
        :param grid_size: grid size for the mesh of the encoder range
        :param ax: plot axis object
        :return:
        """
        # Parametric representation of surface
        u, v = np.mgrid[a:b:grid_size * 1j, a:b:grid_size * 1j]
        x3 = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                x0 = np.column_stack([u[i, j], v[i, j]])
                x0 = torch.tensor(x0, dtype=torch.float32)
                xx = self.metric_tensor(x0)
                xx = torch.linalg.det(xx).detach().numpy()
                x3[i, j] = np.sqrt(xx[0])
        # set up the axes for the first plot
        # fig = plt.figure(figsize=(7, 7))
        # ax = plt.axes(projection='3d')
        if ax is not None:
            # fig = plt.figure(figsize=(7, 7))
            # ax = plt.axes(projection='3d')
            ax.plot_surface(u, v, x3, cmap='viridis', edgecolor='none', alpha=0.5)
            ax.set_title("NN det g")
            # plt.show()
            return None
        else:
            return u, v, x3

    # TODO: some inefficiency here: the u,v coords don't change for the volume plot, so don't store 'em!
    def _store_volume_plot(self):
        # After every K training epochs, generate and store the plot
        u, v, x3 = self.plot_det_g(self.lb, self.ub, self.grid_size)
        self.volume_plots.append((u, v, x3))

    def _store_surface_plot(self):
        # After every K training epochs, generate and store the plot
        u, v, x3 = self.plot_surface(self.lb, self.ub, self.grid_size, None)
        self.surface_plots.append((u, v, x3))

    # TODO: if passing volume plots, use the attribute lb and ub for intrinsic a and b; otherwise need a,b in R^D.
    def animate_fits(self, manifold, training_plots, epochs, a=-1, b=1):
        """
        Animate the chart's learning process.

        :param manifold:
        :param training_plots:
        :param epochs:
        :param a: lower limit of x and y axis
        :param b: upper limits of x and y axis
        :return:
        """
        x1 = training_plots[0][0]
        y1 = training_plots[0][1]
        z1 = training_plots[0][2]

        def update_plot(frame_number, zarray, plot):
            x1 = zarray[frame_number][0]
            y1 = zarray[frame_number][1]
            z1 = zarray[frame_number][2]
            plot[0].remove()
            plot[0] = ax.plot_surface(x1, y1, z1, cmap="magma")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot = [ax.plot_surface(x1, y1, z1, color='0.75', rstride=1, cstride=1)]
        frn = int(epochs / self.fps)  # frame number of the animation
        tn1 = 2.5 * 1000
        ax.set_xlim(a, b)
        ax.set_ylim(a, b)
        ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(training_plots, plot), interval=tn1 / frn)
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Construct the file path with the timestamp
        fn = "plots/autoencoder_plots/" + manifold + "/" + self.act_fun + "_" + timestamp + ".gif"
        # ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
        # ani.save(fn+'.gif',writer='imagemagick',fps=fps)
        ani.save(fn, writer='pillow', fps=self.fps)
        plt.show()

    def plot_surface_and_vol(self, X):
        """
        Wrapper for convenientily plotting the surface and volume density

        :param X: point cloud
        :return: None
        """
        # Plot surface
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, alpha=0.1)
        # TODO probably do not want to use [lb, ub] which are used for the Activation Range
        # Instead you want to be able to see the x and y limits in the ambient space....
        self.plot_surface(self.lb, self.ub, self.grid_size, ax)
        plt.show()
        # Plot volume density
        fig2 = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        self.plot_det_g(self.lb, self.ub, self.grid_size, ax)
        plt.show()


if __name__ == "__main__":
    # Test every activation function and find the one that produces the lowest MSE
    # on the testing set. Here we use a sphere
    #
    # A few runs show no consistent winner.
    from SyntheticData import uniform_sphere
    import time

    # parameters for data
    num_pts = 100
    # Parameters for Auto-Encoder
    extrinsic_dim = 3
    hidden_dim = 16
    num_layers = 0
    # Training parameters
    train_percent = 0.9
    epochs = 5000
    lr = 0.001
    reg = 0.
    orthog_reg = 0
    tn = 0.2
    ntime = 1000
    n_train = int(train_percent * num_pts)
    # Generate synthetic data: hypersphere
    point_cloud = uniform_sphere(num_pts, extrinsic_dim)
    train_data = point_cloud[:n_train, :]
    test_data = point_cloud[n_train:, :]

    # Hyperspheres have d=D-1. Declare an auto-encoder
    intrinsic_dim = extrinsic_dim - 1
    test_losses = []
    for act in ACTIVATION_FUNCTIONS.keys():
        if act != "id" and act != "softmax":
            print(act)
            ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dim, num_layers, act)
            start = time.time()
            ae.fit(train_data, lr, epochs, reg, orthog_reg=orthog_reg)
            end = time.time()
            print("Training time = " + str(end - start))
            # Compute test MSE
            test_loss = ae.loss(test_data, reg).detach().numpy()
            print("\nTest MSE = " + str(test_loss))
            test_losses.append(test_loss)

            # Testing various methods
            x0 = test_data[0, :]
            u0 = ae.encoder(x0)
            print("\nMetric tensor at a point:")
            print(ae.metric_tensor(ae.encoder(x0)))

            print("\nExtrinsic drift via Ito's lemma and auto-diff")
            mu0 = ae.extrinsic_drift_itos(x0)
            print(mu0)

            print("Intrinsic drift via Auto Diff")
            print(ae.euler_intrinsic(u0)[0])

            print("Intrinsic Drift via Linear Solution")
            print(ae.solve_for_int_drift(x0, mu0))

            # Simulating a Brownian path
            u0 = ae.encoder(x0).detach().requires_grad_(True)
            bm = ae.brownian_motion(u0, tn, ntime)
            bm = ae.decoder(bm).detach().numpy()
            # Plot point cloud and sample path
            fig = plt.figure()
            ax = plt.subplot(projection="3d")
            ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], alpha=0.5)
            ax.plot3D(bm[:, 0], bm[:, 1], bm[:, 2], color="black")
            plt.title(act)
            plt.savefig('plots/autoencoder_plots/sphere/' + act + ".png")
    plt.show()
    # Find the lowest test loss
    test_losses = np.array(test_losses)
    acts = list(ACTIVATION_FUNCTIONS.keys())
    min_act = acts[np.argmin(test_losses)]
    print("Lowest test MSE activation = " + str(min_act))
