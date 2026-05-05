"""
ODE system solvers
==================
Solve initial-value problems for a system of N first-order ODEs:

    X'(t) = F(t, X),   X(t0) = X0

where X and F are vectors of length N.

Call signature
--------------
    solver(F, t0, tn, X0, h) -> (t, X)

Parameters
----------
F   : callable(t, X) -> np.ndarray of shape (N,)
      Vector RHS.  **Note:** the labs used a convention where index 0 of
      the returned array is a dummy zero.  This implementation uses plain
      0-based indexing: F(t, X)[0] is the derivative of X[0].
t0  : float
tn  : float
X0  : array-like of shape (N,)
h   : float

Returns
-------
t : np.ndarray, shape (n+1,)
X : np.ndarray, shape (N, n+1)   — X[i] is the i-th variable over time
"""

import math
import numpy as np


def _make_grid(t0, tn, h):
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    return n, t


def euler_system(F, t0, tn, X0, h):
    """Explicit Euler for a system of N first-order ODEs.

    X_{i+1} = X_i + h · F(t_i, X_i)
    """
    X0 = np.asarray(X0, dtype=float)
    N = len(X0)
    n, t = _make_grid(t0, tn, h)
    X = np.zeros((N, n + 1))
    X[:, 0] = X0
    for i in range(n):
        dX = np.asarray(F(t[i], X[:, i]), dtype=float)
        X[:, i + 1] = X[:, i] + h * dX
    return t, X


def runge_kutta2_system(F, t0, tn, X0, h):
    """Runge-Kutta 2nd-order (midpoint) for a system of N ODEs.

    k1 = F(t_i,       X_i)
    k2 = F(t_i + h/2, X_i + h·k1/2)
    X_{i+1} = X_i + h·k2
    """
    X0 = np.asarray(X0, dtype=float)
    N = len(X0)
    n, t = _make_grid(t0, tn, h)
    X = np.zeros((N, n + 1))
    X[:, 0] = X0
    for i in range(n):
        k1 = np.asarray(F(t[i], X[:, i]), dtype=float)
        k2 = np.asarray(F(t[i] + h / 2, X[:, i] + h * k1 / 2), dtype=float)
        X[:, i + 1] = X[:, i] + h * k2
    return t, X


def runge_kutta4_system(F, t0, tn, X0, h):
    """Classic Runge-Kutta 4th-order for a system of N ODEs.

    k1 = F(t_i,       X_i)
    k2 = F(t_i + h/2, X_i + h·k1/2)
    k3 = F(t_i + h/2, X_i + h·k2/2)
    k4 = F(t_i + h,   X_i + h·k3)
    X_{i+1} = X_i + (k1 + 2k2 + 2k3 + k4)·h/6
    """
    X0 = np.asarray(X0, dtype=float)
    N = len(X0)
    n, t = _make_grid(t0, tn, h)
    X = np.zeros((N, n + 1))
    X[:, 0] = X0
    for i in range(n):
        k1 = np.asarray(F(t[i], X[:, i]), dtype=float)
        k2 = np.asarray(F(t[i] + h / 2, X[:, i] + h * k1 / 2), dtype=float)
        k3 = np.asarray(F(t[i] + h / 2, X[:, i] + h * k2 / 2), dtype=float)
        k4 = np.asarray(F(t[i] + h,     X[:, i] + h * k3), dtype=float)
        X[:, i + 1] = X[:, i] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
    return t, X
