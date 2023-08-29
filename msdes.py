import sympy as sp
import numpy as np
from numpy.random import default_rng


def lift_path(xt, f, D):
    ntime = xt.shape[0]-1
    # We need to lift the motion back to the ambient space
    yt = np.zeros((ntime + 1, D))
    for i in range(ntime + 1):
        yt[i, :] = f(xt[i, :]).reshape(D)
    return yt


def euler_maruyama(x0, tn, drift, diffusion, ntime, noise_dim=2, sys_dim=None):
    """ Assumes x0, mu are vectors and b is matrix such that bb^T is the covariance matrix.

    Parameters:
        x0: initial point in N dimensional space.
        tn: the length of time for the simulation.
        drift: the drift function of mu(x).
        diffusion: diffusion coefficient, function of b(x) must be N x d.
        ntime: the number of time sub-intervals in the approximate solution.
        noise_dim: number of Brownian motions.
        sys_dim: the dimensions of system.

    Returns:
        An array of shape (n+1, m) representing the approximate solution to the SDE.
    """
    rng = default_rng()
    if sys_dim is None:
        sys_dim = x0.shape[0]
    h = tn / ntime
    x = np.zeros((ntime + 1, sys_dim))
    x[0, :] = x0
    for i in range(ntime):
        z = rng.normal(scale=np.sqrt(h), size=noise_dim)
        x[i + 1, :] = x[i, :] + drift(x[i, :]) * h + diffusion(x[i, :]).dot(z)
    return x


def metric_tensor(xi, x):
    j = sp.simplify(xi.jacobian(x))
    g = j.T * j
    return sp.simplify(g)


def matrix_divergence(A : sp.Matrix, x):
    n,m = A.shape
    d = sp.zeros(n,1)
    for i in range(n):
        for j in range(m):
            d[i] += sp.simplify(sp.diff(A[i, j], x[j]))
    return sp.simplify(d)


def coefficients(g: sp.Matrix, x):
    ginv = g.inv()
    ginv = sp.simplify(ginv)
    diffusion = ginv.cholesky(hermitian=False)
    diffusion = sp.simplify(diffusion)
    detg = g.det()
    sqrt_detg = sp.sqrt(detg)
    drift = 0.5*(1./sqrt_detg)*matrix_divergence(sp.simplify(sqrt_detg*ginv), x)
    drift = sp.simplify(drift)
    return drift, diffusion


# Creating surfaces via the parameterizations
def surf_param(param, coord, grid, aux=None, p=None):
    """ Compute a mesh of a surface via parameterization. The argument
    'grid' must be a tuple of arrays returned from 'np.mgrid' which the user
    must supply themselves, since boundaries and resolutions are use-case depdendent.
    The tuple returned can be unpacked and passed to plot_surface

    (Parameters):
    param: sympy object defining parameters
    coord: sympy object defining the coordinate transformation
    grid: tuple of the arrays returned from np.mgrid[...]
    aux: sympy Matrix for auxiliary parameters in the metric tensor
    p: numpy array for the numerical values of any auxiliary parameters in the equations

    """
    d = len(grid)
    m = grid[0].shape[0]
    N = coord.shape[0]
    if aux is None:
        coord_np = sp.lambdify([param], coord)
    else:
        coord_np = sp.lambdify([param, aux], coord)

    xx = np.zeros((N, grid[0].shape[0], grid[0].shape[1]))
    for i in range(m):
        for j in range(m):
            w = np.zeros(d)
            for l in range(d):
                w[l] = grid[l][i, j]
            for l in range(N):
                if aux is None:
                    xx[l][i, j] = coord_np(w)[l, 0]
                else:
                    xx[l][i, j] = coord_np(w, p)[l, 0]
    if N == 3:
        x = xx[0]
        y = xx[1]
        z = xx[2]
        return x, y, z
    elif N == 2:
        x = xx[0]
        y = xx[1]
        return x, y
    else:
        return xx


if __name__ == "__main__":
    theta, phi = sp.symbols("theta phi", real=True)
    x = sp.sin(theta) * sp.cos(phi)
    y = sp.sin(theta) * sp.sin(phi)
    z = sp.cos(theta)
    xi = sp.Matrix([x, y, z])
    coord = sp.Matrix([theta, phi])
    g = metric_tensor(xi, coord)
    g = sp.simplify(g)
    mu, Sigma = coefficients(g, coord)

    print(g)
    print(mu)
    print(Sigma)
