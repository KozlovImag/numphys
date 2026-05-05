"""
Finite-difference methods for boundary value problems
=====================================================
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """Solve a tridiagonal system using the Thomas (TDMA) algorithm.

    System:  a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]

    Parameters
    ----------
    a : array-like, lower diagonal  (a[0] is ignored)
    b : array-like, main diagonal
    c : array-like, upper diagonal  (c[-1] is ignored)
    d : array-like, right-hand side

    Returns
    -------
    x : np.ndarray — solution vector
    """
    n = len(b)
    a, b, c, d = map(lambda v: np.array(v, dtype=float), [a, b, c, d])
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x  = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def finite_diff_linear(f_coeffs, t0, tn, y0, yn, h):
    """Finite-difference method for the **linear** BVP

        p(t)·y'' + q(t)·y' + r(t)·y = g(t),
        y(t0) = y0,  y(tn) = yn

    using central differences and the Thomas algorithm.

    Parameters
    ----------
    f_coeffs : callable(t) -> (p, q, r, g)
        Returns the four coefficient functions at a given t.
    t0, tn   : float
    y0, yn   : float  — boundary values
    h        : float  — grid spacing

    Returns
    -------
    t : np.ndarray
    y : np.ndarray  — solution (interior + boundary points)
    """
    import math
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    m = n - 1  # number of interior points

    a_diag = np.zeros(m)
    b_diag = np.zeros(m)
    c_diag = np.zeros(m)
    rhs    = np.zeros(m)

    for k in range(m):
        tk = t[k + 1]
        p, q, r, g = f_coeffs(tk)
        a_diag[k] = p / h**2 - q / (2 * h)
        b_diag[k] = -2 * p / h**2 + r
        c_diag[k] = p / h**2 + q / (2 * h)
        rhs[k] = g

    # Apply boundary conditions
    rhs[0]  -= a_diag[0] * y0
    rhs[-1] -= c_diag[-1] * yn

    y_inner = thomas_algorithm(a_diag, b_diag, c_diag, rhs)
    y = np.concatenate([[y0], y_inner, [yn]])
    return t, y


def simple_iterations(f_nonlin, f_lin_coeffs, t0, tn, y0, yn, h,
                       tol=1e-5, max_iter=500):
    """Finite-difference method for nonlinear BVPs via simple iterations.

    Linearises around the current approximation and repeatedly solves the
    resulting linear system until convergence.

    Parameters
    ----------
    f_nonlin    : callable(t, y, dy) -> float
                  Full nonlinear RHS of y'' = f_nonlin(t, y, y').
    f_lin_coeffs: callable(t, y_prev) -> (p, q, r, g)
                  Linearised coefficients for ``finite_diff_linear`` given
                  the previous iterate y_prev(t).
    t0, tn      : float
    y0, yn      : float
    h           : float
    tol         : float — convergence criterion (max absolute change)
    max_iter    : int

    Returns
    -------
    t     : np.ndarray
    y     : np.ndarray
    n_iter: int — number of iterations performed
    """
    import math
    n_pts = math.ceil((tn - t0) / h) + 1
    t = np.linspace(t0, tn, n_pts)
    y = np.linspace(y0, yn, n_pts)  # initial guess: linear

    for it in range(max_iter):
        y_interp = lambda ti: np.interp(ti, t, y)  # noqa: E731

        def coeff_wrap(ti):
            return f_lin_coeffs(ti, y_interp(ti))

        _, y_new = finite_diff_linear(coeff_wrap, t0, tn, y0, yn, h)
        if np.max(np.abs(y_new - y)) < tol:
            return t, y_new, it + 1
        y = y_new

    return t, y, max_iter
