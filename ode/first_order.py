"""
First-order ODE solvers
=======================
All solvers share the same call signature and return signature so they can
be passed as callables to analysis helpers.

Call signature
--------------
    solver(f, t0, tn, x0, h) -> (t, x)

Parameters
----------
f   : callable(t, x) -> float
      Right-hand side of  x' = f(t, x)
t0  : float  — initial time
tn  : float  — end time
x0  : float  — initial value x(t0)
h   : float  — step size

Returns
-------
t : np.ndarray, shape (n+1,)
x : np.ndarray, shape (n+1,)
"""

import math
import numpy as np


def explicit_euler(f, t0, tn, x0, h):
    """Explicit (forward) Euler method — 1st-order accuracy.

    The simplest one-step method.  Conditionally stable: requires small h
    for stiff problems.
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        x[i + 1] = x[i] + h * f(t[i], x[i])
    return t, x


def implicit_euler(f_implicit, t0, tn, x0, h):
    """Implicit (backward) Euler method — 1st-order accuracy.

    More stable than the explicit variant.  Requires that the user supplies
    a pre-solved closed-form expression for x_{k+1} via ``f_implicit``.

    Parameters
    ----------
    f_implicit : callable(t_next, x_prev, h) -> float
        Closed-form solution of  x_{k+1} = x_k + h·f(t_{k+1}, x_{k+1})
        for x_{k+1} given x_k and h.  Problem-specific.
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        x[i + 1] = f_implicit(t[i + 1], x[i], h)
    return t, x


def runge_kutta2(f, t0, tn, x0, h):
    """Explicit Runge-Kutta, 2nd-order (midpoint / Heun variant).

    Uses the midpoint predictor:
        k1 = f(t_i,       x_i)
        k2 = f(t_i + h/2, x_i + h·k1/2)
        x_{i+1} = x_i + h·k2
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        k1 = f(t[i], x[i])
        k2 = f(t[i] + h / 2, x[i] + h * k1 / 2)
        x[i + 1] = x[i] + h * k2
    return t, x


def runge_kutta4(f, t0, tn, x0, h):
    """Classic Runge-Kutta, 4th-order — the workhorse of numerical ODEs.

    Coefficients:
        k1 = f(t_i,       x_i)
        k2 = f(t_i + h/2, x_i + h·k1/2)
        k3 = f(t_i + h/2, x_i + h·k2/2)
        k4 = f(t_i + h,   x_i + h·k3)
        x_{i+1} = x_i + (k1 + 2k2 + 2k3 + k4)·h/6
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    for i in range(n):
        k1 = f(t[i], x[i])
        k2 = f(t[i] + h / 2, x[i] + h * k1 / 2)
        k3 = f(t[i] + h / 2, x[i] + h * k2 / 2)
        k4 = f(t[i] + h,     x[i] + h * k3)
        x[i + 1] = x[i] + (k1 + 2*k2 + 2*k3 + k4) * h / 6
    return t, x


def adams2_explicit(f, t0, tn, x0, h):
    """Explicit Adams-Bashforth, 2nd-order (two-step multistep method).

    Uses RK2 to generate the first step, then:
        x_{i+1} = x_i + h·(3f_i - f_{i-1}) / 2

    Slightly more accurate per step than Euler for the same h, but
    requires storing the previous derivative value.
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    # Bootstrap first step with RK2
    k1 = f(t[0], x[0])
    k2 = f(t[0] + h / 2, x[0] + h * k1 / 2)
    x[1] = x[0] + h * k2
    f_prev = k1
    for i in range(1, n):
        f_curr = f(t[i], x[i])
        x[i + 1] = x[i] + h * (3*f_curr - f_prev) / 2
        f_prev = f_curr
    return t, x


def adams2_implicit(f_implicit_adams, t0, tn, x0, h):
    """Implicit Adams-Moulton, 2nd-order (trapezoidal rule predictor-corrector).

    x_{i+1} = x_i + h·(f_{i+1} + f_i) / 2

    Requires a problem-specific closed-form inversion.

    Parameters
    ----------
    f_implicit_adams : callable(t_next, x_prev, h, f_prev_val, f_curr_val, step)
        Returns x_{k+1}.  ``step`` = 0 at t0 (uses plain RHS),
        = 1 at the first step (uses implicit Euler), >= 2 uses Adams formula.
    """
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    x = np.zeros(n + 1)
    x[0] = x0
    f_vals = np.zeros(n + 1)
    f_vals[0] = f_implicit_adams(t[0], x[0], h, 0, 0, 0)
    for i in range(n):
        x[i + 1] = f_implicit_adams(
            t[i + 1], x[i], h,
            f_vals[i], f_vals[max(i-1, 0)], i + 1
        )
        f_vals[i + 1] = f_implicit_adams(t[i + 1], x[i + 1], h, 0, 0, 0)
    return t, x
