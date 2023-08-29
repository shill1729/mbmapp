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
import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import torch

from msdes import metric_tensor, coefficients, surf_param, euler_maruyama, lift_path
from autoencoder import AutoEncoder


@st.cache_resource
def symbolic_computations(coordinate, chart_map):
    inputs = sp.Matrix([sp.Symbol(s.strip()) for s in coordinate.split(",")])
    manifold = sp.Matrix([sp.sympify(s.strip()) for s in chart_map.split(",")])
    # Compute the geometry and SDE
    g = metric_tensor(manifold, inputs)
    mu, Sigma = coefficients(g, inputs)
    # Lambdify the SymPy functions for the SDE coefficients
    mu_np = sp.lambdify([inputs], mu)
    Sigma_np = sp.lambdify([inputs], Sigma)
    # Get an numerical map back to the manifold from the chart
    f = sp.lambdify([inputs], manifold)
    return g, mu, Sigma, mu_np, Sigma_np, f, inputs, manifold


def plot_ground_truth(a, b, c, d, inputs, manifold, npaths, ntime, intrinsic_dim, extrinsic_dim,
                      x0, tn, mu_np, Sigma_np, f):
    # Button to generate first plot
    if st.button('Generate ground truth plot'):
        # Generate new data for the plot (replace this with your plot generation logic)
        # Computing a grid for the surface
        # TODO need to handle planar cases and curves in space.
        u = np.linspace(a, b, 50)
        v = np.linspace(c, d, 50)
        grid = np.meshgrid(u, v, indexing="ij")
        surf = surf_param(inputs, manifold, grid)

        # Generating ensembles of sample paths in the local space and ambient space
        local_paths = np.zeros((npaths, ntime + 1, intrinsic_dim))
        ambient_paths = np.zeros((npaths, ntime + 1, extrinsic_dim))
        for i in range(npaths):
            xt = euler_maruyama(x0, tn, lambda x: mu_np(x).reshape(intrinsic_dim), Sigma_np, ntime)
            local_paths[i] = xt
            ambient_paths[i] = lift_path(xt, f, extrinsic_dim)
        st.session_state.ground_truth = surf, local_paths, ambient_paths
        # TODO this is some times throwing a warning although I cannot figure why (about it not being initialized)
        st.session_state.npaths = npaths

    # Now check for the session variable and plot
    if st.session_state.ground_truth is not None:
        surf, local_paths, ambient_paths = st.session_state.ground_truth
        # Create a single figure with two horizontally-arranged subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # First plot: the surface mesh and the ambient sample paths
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(surf[0], surf[1], surf[2], alpha=0.5, cmap="viridis")
        for i in range(st.session_state.npaths):
            yt = ambient_paths[i]
            ax1.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], c="black", alpha=0.5)
        # Second plot: the chart and local sample paths
        # Define the vertices of the chart as a rectangle [a,b]x[c,d]
        x1 = [a, b, b, a, a]
        y1 = [c, c, d, d, c]
        # Plot the perimeter of the rectangle
        ax2.plot(x1, y1)
        # Plot the local Brownian motions
        for i in range(st.session_state.npaths):
            yt = local_paths[i]
            ax2.plot(yt[:, 0], yt[:, 1], c="black", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)


def generate_point_cloud(extrinsic_dim, intrinsic_dim, num_pts, a, b, c, d, f):
    X = np.random.rand(num_pts, intrinsic_dim)
    X[:, 0] = (b - a) * X[:, 0] + a
    X[:, 1] = (c - d) * X[:, 1] + d
    point_cloud = lift_path(X, f, extrinsic_dim)
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
    return x, y, z, point_cloud


# @st.cache_resource
def load_model(extrinsic_dim, intrinsic_dim, hidden_dim, num_layers):
    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dim, num_layers, "tanh")
    return ae


def model_output(npaths, tn, ntime, intrinsic_dim, extrinsic_dim, _ae, _point_cloud, x, y, z):
    # Generate Brownian motion paths from the model
    local_paths_model = np.zeros((npaths, ntime + 1, intrinsic_dim))
    ambient_paths_model = np.zeros((npaths, ntime + 1, extrinsic_dim))
    for i in range(npaths):
        # xt = euler_maruyama(x0, tn, lambda x:mu_np(x).reshape(2), Sigma_np, ntime)
        xt = _ae.brownian_motion(torch.zeros(intrinsic_dim, requires_grad=True), tn, ntime, None)
        local_paths_model[i] = xt.detach()
        ambient_paths_model[i] = _ae.decoder(xt).detach()

    q = _ae.forward(_point_cloud).detach()
    w = _ae.encoder(_point_cloud).detach()
    # Plot learned surface, point cloud and embedded point cloud colored
    # by x-axis values, next to the 2d plot of the encoded space in (-1,1)^2.
    # Create a single figure with two horizontally-arranged subplots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # First plot (3D scatter)
    ax1 = fig2.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(x, y, z, alpha=0.2)
    ax1.scatter(q[:, 0], q[:, 1], q[:, 2], c=q[:, 0])
    for i in range(npaths):
        yt = ambient_paths_model[i]
        ax1.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], c="black", alpha=0.5)
    _ae.plot_surface(-1., 1., 50, ax1)
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
    st.pyplot(fig2)


def main():
    # ================================================================================================
    # 1. Input:
    # ================================================================================================
    # Sidebar Inputs
    # User input for parameterization
    coordinate = st.sidebar.text_input("Enter local coordinates (comma-separated)", "x, y")
    chart_map = st.sidebar.text_input("Enter chart mapping formula (comma-separated)", "x,y,x**2+y**2")
    # User input for grid bounds
    grid_bds_input = st.sidebar.text_input("Grid bounds a,b,c,d (comma-separated)", "-1, 1, -1, 1")

    # User input for path parameters
    x0_input = st.sidebar.text_input("Initial point (comma-separated)", "0, 0")  # This is the initial point
    # for the true paths only. We will use the origin for the model paths since we only tanh here.
    tn = st.sidebar.number_input("Time horizon (tn)", value=1., min_value=0.01, step=0.01)
    npaths = st.sidebar.number_input("Number of paths", value=1, min_value=1)
    ntime = st.sidebar.number_input("Number of time steps", value=10000, min_value=50)

    # User input for point cloud
    num_pts = st.sidebar.number_input("Number of points in point cloud", value=100, min_value=1)
    # User input for autoencoder
    hidden_dim = st.sidebar.number_input("Hidden layer dimension", value=3, min_value=1)
    num_layers = st.sidebar.number_input("Number of (inner) layers", value=1, min_value=0)
    lr = st.sidebar.number_input("Learning rate", value=0.01, min_value=0.0001, step=0.001, format="%.4f")
    epochs = st.sidebar.number_input("Number of epochs", value=5000, min_value=1)
    # Regularization parameters
    ctr_reg = st.sidebar.number_input("Contractive reg.", value=0., min_value=0., step=0.0001, format="%.5f")
    sparse_reg = st.sidebar.number_input("Sparse reg.", value=0., min_value=0., step=0.0001, format="%.5f")
    orthog_reg = st.sidebar.number_input("Orthogonal reg.", value=0., min_value=0., step=0.0001, format="%.5f")
    diffeo_reg = st.sidebar.number_input("Diffeo. reg.", value=0., min_value=0., step=0.0001, format="%.5f")

    # ================================================================================================
    # 2. Symbolic computation of SDE coefficients:
    # ================================================================================================
    # Cache resources for our symbolic computations.
    g, mu, Sigma, mu_np, Sigma_np, f, inputs, manifold = symbolic_computations(coordinate, chart_map)

    # Print the equations--always! But the cache prevents recomputing them.
    st.write("Metric tensor")
    st.write(g)
    st.write("Local chart BM drift:")
    st.write(mu)
    st.write("Local chart BM diffusion:")
    st.write(Sigma)

    # ================================================================================================
    # 3. Simulate n paths using the exact dynamics on the exact manifold.
    # ================================================================================================
    # Initialize session state variables for the plots
    if 'ground_truth' not in st.session_state:
        st.session_state.ground_truth = None
    if 'npaths' not in st.session_state:
        st.session_state.npaths = npaths

    grid_bds = [float(s.strip()) for s in grid_bds_input.split(",")]
    x0 = np.array([float(s.strip()) for s in x0_input.split(",")])
    a = grid_bds[0]
    b = grid_bds[1]
    c = grid_bds[2]
    d = grid_bds[3]
    extrinsic_dim = manifold.shape[0]
    intrinsic_dim = inputs.shape[0]

    # Plot the ground truth
    plot_ground_truth(a, b, c, d, inputs, manifold, npaths, ntime, intrinsic_dim, extrinsic_dim,
                      x0, tn, mu_np, Sigma_np, f)

    # ================================================================================================
    # 4. Generate a point cloud
    # ================================================================================================
    # Generate a point cloud:
    # Two ways to generate point clouds.
    # Naive way: uniform parameters U,V on [a,b]x[c,d] and take the point cloud
    # X = f(U, V) where f is the chart from open set in R^d to M in R^D.
    # Expensive way: Randomly sample from Brownian motion paths for long enough T.
    if "gen_cloud" not in st.session_state:
        st.session_state.gen_cloud = True
    if "num_pts" not in st.session_state:
        st.session_state.num_pts = num_pts
    if "point_cloud" not in st.session_state:
        st.session_state.point_cloud = None
    if "chart" not in st.session_state:
        st.session_state.chart = f
    if 'grid_bds' not in st.session_state:
        st.session_state.grid_bds = grid_bds
    if num_pts != st.session_state.num_pts:
        st.session_state.num_pts = num_pts
        st.session_state.gen_cloud = True
    if f != st.session_state.chart:
        st.session_state.chart = f
        st.session_state.gen_cloud = True
    if grid_bds != st.session_state.grid_bds:
        st.session_state.grid_bds = grid_bds
        st.session_state.gen_cloud = True
    if st.session_state.gen_cloud:
        a = st.session_state.grid_bds[0]
        b = st.session_state.grid_bds[1]
        c = st.session_state.grid_bds[2]
        d = st.session_state.grid_bds[3]
        x, y, z, point_cloud = generate_point_cloud(extrinsic_dim, intrinsic_dim,
                                                    st.session_state.num_pts, a, b, c, d, st.session_state.chart)
        st.session_state.gen_cloud = False
        st.session_state.point_cloud = x, y, z, point_cloud

    # if st.session_state.point_cloud is not None:
    #     x, y, z, point_cloud = st.session_state.point_cloud
    #     fig = plt.figure()
    #     ax = plt.subplot(projection="3d")
    #     ax.scatter(x, y, z)
    #     st.pyplot(fig)
    # ================================================================================================
    # 5. Train an auto-encoder
    # ================================================================================================
    ae = AutoEncoder(extrinsic_dim, intrinsic_dim, hidden_dim, num_layers, "tanh")
    if "ae" not in st.session_state:
        st.session_state.ae = ae
    if "extrinsic_dim" not in st.session_state:
        st.session_state.extrinsic_dim = extrinsic_dim
    if "intrinsic_dim" not in st.session_state:
        st.session_state.intrinsic_dim = intrinsic_dim
    if "hidden_dim" not in st.session_state:
        st.session_state.hidden_dim = hidden_dim
    if "num_layers" not in st.session_state:
        st.session_state.num_layers = num_layers

    if extrinsic_dim != st.session_state.extrinsic_dim:
        st.session_state.extrinsic_dim = extrinsic_dim
        st.session_state.ae = AutoEncoder(st.session_state.extrinsic_dim,
                                          st.session_state.intrinsic_dim,
                                          st.session_state.hidden_dim,
                                          st.session_state.num_layers
                                          )
    if intrinsic_dim != st.session_state.intrinsic_dim:
        st.session_state.intrinsic_dim = intrinsic_dim
        st.session_state.ae = AutoEncoder(st.session_state.extrinsic_dim,
                                          st.session_state.intrinsic_dim,
                                          st.session_state.hidden_dim,
                                          st.session_state.num_layers
                                          )
    if hidden_dim != st.session_state.hidden_dim:
        st.session_state.hidden_dim = hidden_dim
        st.session_state.ae = AutoEncoder(st.session_state.extrinsic_dim,
                                          st.session_state.intrinsic_dim,
                                          st.session_state.hidden_dim,
                                          st.session_state.num_layers
                                          )
    if num_layers != st.session_state.num_layers:
        st.session_state.num_layers = num_layers
        st.session_state.ae = AutoEncoder(st.session_state.extrinsic_dim,
                                          st.session_state.intrinsic_dim,
                                          st.session_state.hidden_dim,
                                          st.session_state.num_layers
                                          )

    if "model_output" not in st.session_state:
        st.session_state.model_output = None
    if st.button("Train auto-encoder"):
        x, y, z, ptc = st.session_state.point_cloud

        st.session_state.ae.fit(ptc, lr, epochs,
                                ctr_reg, sparse_reg, orthog_reg, diffeo_reg)
    if st.session_state.point_cloud is not None:
        x, y, z, ptc = st.session_state.point_cloud
        st.write("Training loss = " + str(st.session_state.ae.loss(ptc).detach().numpy()))
    if 'npaths2' not in st.session_state:
        st.session_state.npaths2 = npaths
    if st.button("Generate model paths"):
        # Generate Brownian motion paths from the model
        local_paths_model = np.zeros((npaths, ntime + 1, st.session_state.intrinsic_dim))
        ambient_paths_model = np.zeros((npaths, ntime + 1, st.session_state.extrinsic_dim))
        for i in range(npaths):
            # xt = euler_maruyama(x0, tn, lambda x:mu_np(x).reshape(2), Sigma_np, ntime)
            xt = st.session_state.ae.brownian_motion(torch.zeros(intrinsic_dim, requires_grad=True), tn, ntime, None)
            local_paths_model[i] = xt.detach()
            ambient_paths_model[i] = st.session_state.ae.decoder(xt).detach()
        st.session_state.model_output = local_paths_model, ambient_paths_model
        st.session_state.npaths2 = npaths
    if st.session_state.model_output is not None and st.session_state.point_cloud is not None:
        x, y, z, ptc = st.session_state.point_cloud
        local_paths_model, ambient_paths_model = st.session_state.model_output
        q = st.session_state.ae.forward(ptc).detach()
        w = st.session_state.ae.encoder(ptc).detach()
        # Plot learned surface, point cloud and embedded point cloud colored
        # by x-axis values, next to the 2d plot of the encoded space in (-1,1)^2.
        # Create a single figure with two horizontally-arranged subplots
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # First plot (3D scatter)
        ax1 = fig2.add_subplot(1, 2, 1, projection="3d")
        ax1.scatter(x, y, z, alpha=0.2)
        ax1.scatter(q[:, 0], q[:, 1], q[:, 2], c=q[:, 0])
        for i in range(st.session_state.npaths2):
            yt = ambient_paths_model[i]
            ax1.plot3D(yt[:, 0], yt[:, 1], yt[:, 2], c="black", alpha=0.5)
        st.session_state.ae.plot_surface(-1., 1., 50, ax1)
        # Second plot (2D scatter)
        # Define the vertices of the square
        x1 = [-1, 1, 1, -1, -1]
        y1 = [-1, -1, 1, 1, -1]
        # Plot the perimeter of the square
        ax2.plot(x1, y1)
        ax2.scatter(w[:, 0], w[:, 1], c=q[:, 0])
        # Plot the local Brownian motions
        for i in range(st.session_state.npaths2):
            yt = local_paths_model[i]
            ax2.plot(yt[:, 0], yt[:, 1], c="black", alpha=0.5)
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        plt.tight_layout()
        st.pyplot(fig2)


main()