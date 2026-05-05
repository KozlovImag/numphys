"""
2-D diffusion / reaction-diffusion PDE solver
=============================================
Solves the 2-D PDE

    ∂u/∂t = D(x,y) · (∂²u/∂x² + ∂²u/∂y²) + S(t, x, y, u)

on a rectangular domain [x0,xn] × [y0,yn], t ∈ [t0, tn], using an explicit
(forward Euler in time, central differences in space) finite-difference scheme.

Stability condition (CFL):  h_t ≤ l² / (4 · max(D))
"""

import math
import numpy as np


def diffusion2d(
    D, S, u0,
    t0, tn, h_t,
    x0, xn, y0, yn, l,
    bc_left, bc_right, bc_bottom, bc_top,
    save_every=1,
):
    """Explicit finite-difference solver for a 2-D diffusion-reaction PDE.

    Parameters
    ----------
    D         : callable(x, y) -> float or array
                Diffusion coefficient (may be spatially varying).
    S         : callable(t, x, y, u) -> array
                Source / sink term.
    u0        : callable(x, y) -> np.ndarray  — initial condition.
    t0, tn    : float  — time interval.
    h_t       : float  — time step.
    x0, xn   : float  — x-domain boundaries.
    y0, yn   : float  — y-domain boundaries.
    l         : float  — spatial grid step (same in x and y).
    bc_left   : callable(t, y) -> float  — left   boundary (x = x0).
    bc_right  : callable(t, y) -> float  — right  boundary (x = xn).
    bc_bottom : callable(t, x) -> float  — bottom boundary (y = y0).
    bc_top    : callable(t, x) -> float  — top    boundary (y = yn).
    save_every: int  — store every n-th time slice to save memory.

    Returns
    -------
    T : np.ndarray, shape (n_saved,)         — saved time points
    X : np.ndarray, shape (nx,)              — x grid
    Y : np.ndarray, shape (ny,)              — y grid
    U : list of np.ndarray, shape (nx, ny)   — saved u snapshots
    """
    nx = math.ceil((xn - x0) / l) + 1
    ny = math.ceil((yn - y0) / l) + 1
    nt = math.ceil((tn - t0) / h_t)

    X = np.array([x0 + i * l for i in range(nx)])
    Y = np.array([y0 + j * l for j in range(ny)])
    T_saved = []
    U_saved = []

    # Initial condition
    u = u0(X, Y)

    r = h_t / l**2  # spatial coefficient

    for step in range(nt + 1):
        t = t0 + step * h_t

        if step % save_every == 0:
            T_saved.append(t)
            U_saved.append(u.copy())

        if step == nt:
            break

        u_new = u.copy()

        # Interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                d = D(X[i], Y[j])
                lap = (u[i+1,j] - 2*u[i,j] + u[i-1,j] +
                       u[i,j+1] - 2*u[i,j] + u[i,j-1])
                src = S(t, X[i], Y[j], u[i, j])
                u_new[i, j] = u[i, j] + r * d * lap + h_t * src

        # Boundary conditions
        for j in range(ny):
            u_new[0,  j] = bc_left(t,  Y[j])
            u_new[-1, j] = bc_right(t, Y[j])
        for i in range(nx):
            u_new[i, 0]  = bc_bottom(t, X[i])
            u_new[i, -1] = bc_top(t,   X[i])

        u = u_new

    return np.array(T_saved), X, Y, U_saved
