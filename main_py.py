# Specifications and Requirements of SDE Manifold App (local chart version):
#
# The user can pass strings defining
#
# 1. the local coordinates x,y,z,u,w,… in d-space
# 2. the chart mapping form (x,y,z,…) to phi(x,y,z,…) in D-space
# 3. Optional potential function? Late feature
# 4. Extra parameters a,b,c,d,e for coefficients or whatever. Late feature
# 5. Network and training parameters
#
# The program will then
#
# 1. Compute the SDE coefficients of Brownian motion in the local chart
# 2. Simulate N sample paths
# 3. Map them to the ambient space
# 4. plot both paths, with the ambient paths on a surface of the shape.
#
# Train an auto-encoder with diffeomorphic regularization on a uniform point cloud obtained
# from the previous sample path simulation (of their tails). We generate N paths again
# and then plot the encoded point cloud in the (-1,1)^d box and plot the local model paths
# and the learned surface with the ambient model paths
import numpy as np
import sympy as sp
from msdes import metric_tensor, coefficients, surf_param, euler_maruyama, lift_path
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder
import torch
# An app for Manifold Brownian motion and an auto-encoder.
# Input in the streamlit app
# String input for parameterization
coordinate = "x, y"
chart_map = "x, y, x**2+y**2"
# Numeric input for sample paths
x0 = "0., 0."
tn = 5
npaths = 5
ntime = 10000
# Intervals for the grid for the surface mesh are to be given as a string, a,b,c,d for
# [a,b]x[c,d] grid in the plane, or [a,b] for grid on the line.
grid_bds = "-2, 2, -2, 2"
# Point cloud
num_pts = 100
# network parameters:
hidden_dim = 16
num_layers = 1
# Training parameters
lr = 0.001
epochs = 9000
# activation = "tanh" ## Hardcode this for now.
# END OF USER INPUT

# Body for the app.
# Convert from string to sympy
inputs = sp.Matrix([sp.Symbol(s.strip()) for s in coordinate.split(",")])
manifold = sp.Matrix([sp.sympify(s.strip()) for s in chart_map.split(",")])
grid_bds = [float(s.strip()) for s in grid_bds.split(",")]
x0 = np.array([float(s.strip()) for s in x0.split(",")])
# Get the dimensions of the manifold
extrinsic_dim = manifold.shape[0]
intrinsic_dim = inputs.shape[0]
# Compute the geometry and SDE
g = metric_tensor(manifold, inputs)
mu, Sigma = coefficients(g, inputs)
# Lambdify the SymPy functions for the SDE coefficients
mu_np = sp.lambdify([inputs], mu)
Sigma_np = sp.lambdify([inputs], Sigma)
# Get an numerical map back to the manifold from the chart
f = sp.lambdify([inputs], manifold)

# Print the equations
print("Metric tensor")
print(g)
print("Intrinsic BM drift:")
print(mu)
print("Intrinsic BM diffusion:")
print(Sigma)

# Done with SymPy methods now. The rest shall all be numerical..
# Computing a grid for the surface
# TODO need to handle planar cases and curves in space.
u = np.linspace(grid_bds[0], grid_bds[1], 50)
v = np.linspace(grid_bds[2], grid_bds[3], 50)
grid = np.meshgrid(u, v, indexing="ij")
surf = surf_param(inputs, manifold, grid)

# Generating ensembles of sample paths in the local space and ambient space
local_paths = np.zeros((npaths, ntime+1, intrinsic_dim))
ambient_paths = np.zeros((npaths, ntime+1, extrinsic_dim))
for i in range(npaths):
    xt = euler_maruyama(x0, tn, lambda x:mu_np(x).reshape(2), Sigma_np, ntime)
    local_paths[i] = xt
    ambient_paths[i] = lift_path(xt, f, extrinsic_dim)

# Create a single figure with two horizontally-arranged subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# First plot: the surface mesh and the ambient sample paths
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(surf[0], surf[1], surf[2], alpha=0.5, cmap="viridis")
for i in range(npaths):
    yt = ambient_paths[i]
    ax1.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], c="black", alpha=0.5)
# Second plot: the chart and local sample paths
# Define the vertices of the chart as a rectangle [a,b]x[c,d]
a = grid_bds[0]
b = grid_bds[1]
c = grid_bds[2]
d = grid_bds[3]
x1 = [a, b, b, a, a]
y1 = [c, c, d, d, c]
# Plot the perimeter of the rectangle
ax2.plot(x1, y1)
# Plot the local Brownian motions
for i in range(npaths):
    yt = local_paths[i]
    ax2.plot(yt[:, 0], yt[:, 1], c="black", alpha=0.5)
plt.tight_layout()
plt.show()

# Two ways to generate point clouds.
# Naive way: uniform parameters U,V on [a,b]x[c,d] and take the point cloud
# X = f(U, V) where f is the chart from open set in R^d to M in R^D.
# Expensive way: Randomly sample from Brownian motion paths for long enough T.
X = np.random.rand(num_pts, intrinsic_dim)
X[:, 0] = (b-a)*X[:, 0]+a
X[:, 1] = (c-d)*X[:, 1]+d
point_cloud = lift_path(X, f, extrinsic_dim)
x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

# Train an autoencoder on the point cloud

ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dim, num_layers, "tanh")
# TODO: StreamLit button to train the NN--since this takes awhile, we want manual control
ae.fit(point_cloud, lr, epochs)
model_cloud = ae.forward(point_cloud).detach()

xm, ym, zm = model_cloud[:, 0], model_cloud[:, 1], model_cloud[:, 2]
# fig1 = plt.figure()
# ax = plt.subplot(projection="3d")
# ax.scatter(x, y, z)
# ax.scatter(xm, ym, zm)
# plt.legend(["Data", "Model"])
# plt.show()

# Plot the point cloud and color points by x-axis.
q = ae.forward(point_cloud).detach()
w = ae.encoder(point_cloud).detach()
# TODO: a button to simulate model paths since this also takes a while
# Generate Brownian motion paths from the model
local_paths_model = np.zeros((npaths, ntime+1, intrinsic_dim))
ambient_paths_model = np.zeros((npaths, ntime+1, extrinsic_dim))
for i in range(npaths):
    # xt = euler_maruyama(x0, tn, lambda x:mu_np(x).reshape(2), Sigma_np, ntime)
    xt = ae.brownian_motion(torch.zeros(intrinsic_dim, requires_grad=True), tn, ntime, None)
    local_paths_model[i] = xt.detach()
    ambient_paths_model[i] = ae.decoder(xt).detach()


# Plot learned surface, point cloud and embedded point cloud colored
# by x-axis values, next to the 2d plot of the encoded space in (-1,1)^2.
# Create a single figure with two horizontally-arranged subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# First plot (3D scatter)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(x, y, z, alpha=0.2)
ax1.scatter(q[:, 0], q[:, 1], q[:, 2], c=q[:, 0])
for i in range(npaths):
    yt = ambient_paths_model[i]
    ax1.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], c="black", alpha=0.5)
ae.plot_surface(-1.1, 1.1, 50, ax1)
# Second plot (2D scatter)
# Define the vertices of the square
x1 = [-1, 1, 1, -1, -1]
y1 = [-1, -1, 1, 1, -1]
# Plot the perimeter of the square
ax2.plot(x1, y1)
ax2.scatter(w[:, 0], w[:, 1], c=q[:, 0])
# Plot the local Brownian motions
for i in range(npaths):
    yt = local_paths_model[i]
    ax2.plot(yt[:, 0], yt[:, 1], c="black", alpha=0.5)
ax2.set_xlim(-1.1, 1.1)
ax2.set_ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()