"""
Error analysis tools
=====================
"""

import numpy as np
import matplotlib.pyplot as plt


def max_error(solver, f, t0, tn, x0, solution, h_values):
    """Compute the max absolute error vs step size.

    Parameters
    ----------
    solver   : callable(f, t0, tn, x0, h) -> (t, x)
    f        : ODE RHS
    t0, tn   : float
    x0       : initial condition (scalar or array)
    solution : callable(t) -> analytical solution (same shape as x)
    h_values : iterable of float — step sizes to test

    Returns
    -------
    h_arr  : np.ndarray
    err_arr: np.ndarray — max |numerical - analytical| for each h
    """
    h_arr, err_arr = [], []
    for h in h_values:
        t, x = solver(f, t0, tn, x0, h)
        sol = np.asarray(solution(t))
        x = np.asarray(x)
        if sol.ndim > 1:
            err = np.max(np.abs(sol[0] - x[0]))
        else:
            err = np.max(np.abs(sol - x))
        h_arr.append(h)
        err_arr.append(err)
    return np.array(h_arr), np.array(err_arr)


def rms_error(solver, f, t0, tn, x0, solution, h_values):
    """RMS error vs step size.

    Returns
    -------
    h_arr, err_arr
    """
    h_arr, err_arr = [], []
    for h in h_values:
        t, x = solver(f, t0, tn, x0, h)
        sol = np.asarray(solution(t))
        x = np.asarray(x)
        if sol.ndim > 1:
            diff = sol[0] - x[0]
        else:
            diff = sol - x
        err = np.sqrt(np.mean(diff**2))
        h_arr.append(h)
        err_arr.append(err)
    return np.array(h_arr), np.array(err_arr)


def half_step_error(solver, f, t0, tn, x0, h_values):
    """Richardson extrapolation — error estimate without analytical solution.

    Computes  e ≈ |x_h(tn) - x_{h/2}(tn)|  as a proxy for the global error.

    Returns
    -------
    h_arr, err_arr
    """
    h_arr, err_arr = [], []
    for h in h_values:
        _, x1 = solver(f, t0, tn, x0, h)
        _, x2 = solver(f, t0, tn, x0, h / 2)
        x1_end = np.asarray(x1)[..., -1]
        x2_end = np.asarray(x2)[..., -1]
        err = np.max(np.abs(x1_end - x2_end))
        h_arr.append(h)
        err_arr.append(err)
    return np.array(h_arr), np.array(err_arr)


def plot_errors(h_arr, err_arr, label="", ax=None, log_log=True):
    """Plot error vs step size.

    Parameters
    ----------
    h_arr, err_arr : np.ndarray
    label  : str
    ax     : matplotlib Axes or None
    log_log: bool — use log-log scale if True
    """
    if ax is None:
        _, ax = plt.subplots()
    if log_log:
        ax.loglog(h_arr, err_arr, "o-", label=label)
        ax.set_xlabel("Step size h  (log)")
        ax.set_ylabel("Error (log)")
    else:
        ax.plot(h_arr, err_arr, "o-", label=label)
        ax.set_xlabel("Step size h")
        ax.set_ylabel("Error")
    ax.grid(True, which="both")
    if label:
        ax.legend()
    return ax
