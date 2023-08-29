# This module contains smooth activation functions and their derivatives for use in the
# (smooth) auto-encoder neural network for learning manifold geometry
import torch.nn as nn
import torch

# Truncating (-inf, inf) to (-bd, bd)
truncation_bd = 100

ACTIVATION_FUNCTIONS = {
    "id": nn.Identity(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(alpha=2),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    "softplus": nn.Softplus(),
    "gaussian": lambda x: torch.exp(-x ** 2),
    "silu": nn.SiLU(),
    "celu": nn.CELU(alpha=5),
    "softmax": nn.Softmax(dim=1),
    "sine": torch.sin
}

ACTIVATION_RANGES = {
    "id": (-truncation_bd, truncation_bd),
    "tanh": (-1, 1),
    "elu": (-ACTIVATION_FUNCTIONS["elu"].alpha, truncation_bd),  # Assuming alpha = 1
    "sigmoid": (0, 1),
    "gelu": (-0.17, truncation_bd),
    "softplus": (0, truncation_bd),
    "gaussian": (0, 1),
    "silu": (-0.278, truncation_bd),
    "celu": (-ACTIVATION_FUNCTIONS["celu"].alpha, truncation_bd),
    "softmax": (0, 1),
    "sine": (-1, 1)
}


def sigmoid_derivative(x):
    a = ACTIVATION_FUNCTIONS["sigmoid"](x)
    return a * (1 - a)


def gelu_derivative(y):
    nrv = torch.distributions.normal.Normal(0., 1.)
    pi = torch.tensor(torch.pi)
    gelu_deriv = nrv.cdf(y) + y * torch.exp(-y ** 2 / 2) / torch.sqrt(2 * pi)
    return gelu_deriv


def silu_derivative(y):
    silu_deriv = 1 + torch.exp(-y) + y * torch.exp(-y)
    silu_deriv = silu_deriv / (1 + torch.exp(-y)) ** 2
    return silu_deriv


def celu_derivative(y):
    celu_deriv = 1 * (y >= 0) + (y < 0) * torch.exp(y / ACTIVATION_FUNCTIONS["celu"].alpha)
    return celu_deriv


ACTIVATION_DERIVATIVES = {
    "id": lambda y: torch.ones(y.size()),
    "tanh": lambda y: 1 - ACTIVATION_FUNCTIONS["tanh"](y) ** 2,
    "elu": lambda y: 1 * (y > 0) + (y <= 0) * torch.exp(y),
    "sigmoid": sigmoid_derivative,
    "gelu": gelu_derivative,
    "softplus": lambda y: 1 / (1 + torch.exp(-y)),
    "gaussian": lambda y: -2 * y * torch.exp(-y ** 2),
    "silu": silu_derivative,
    "celu": celu_derivative,
    "softmax": None,
    "sine": torch.cos
}
