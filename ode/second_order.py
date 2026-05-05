"""
Second-order ODE solvers
========================
Solve initial-value problems of the form

    y''(t) = f(t, y, y')
    y(t0) = y0,  y'(t0) = dy0

All solvers share the same call signature and return signature.

Call signature
--------------
    solver(f, t0, tn, x0, h) -> (t, x)

Parameters
----------
f   : callable(t, state) -> float
      RHS of the 2nd-order ODE.
      ``state = [y, y']``  (array of length 2).
t0  : float
tn  : float
x0  : array-like of length 2  — [y(t0), y'(t0)]
h   : float

Returns
-------
t : np.ndarray, shape (n+1,)
x : np.ndarray, shape (2, n+1)   — x[0] = y(t), x[1] = y'(t)
"""

import math
import numpy as np


def _make_grid(t0, tn, h):
    n = math.ceil((tn - t0) / h)
    t = np.array([t0 + i * h for i in range(n + 1)])
    return n, t


def explicit_euler2(f, t0, tn, x0, h):
    """Explicit Euler for 2nd-order ODEs.

    y_{i+1}  = y_i  + h · y'_i
    y'_{i+1} = y'_i + h · f(t_i, [y_i, y'_i])
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    for i in range(n):
        acc = f(t[i], x[:, i])
        x[0, i + 1] = x[0, i] + h * x[1, i]
        x[1, i + 1] = x[1, i] + h * acc
    return t, x


def semi_implicit_euler2(f, t0, tn, x0, h):
    """Semi-implicit (symplectic) Euler for 2nd-order ODEs.

    First updates velocity with the old position, then updates position
    with the *new* velocity — conserves a modified energy better than
    the explicit variant.

    y'_{i+1} = y'_i + h · f(t_i, [y_i, y'_i])
    y_{i+1}  = y_i  + h · y'_{i+1}
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    for i in range(n):
        acc = f(t[i], x[:, i])
        x[1, i + 1] = x[1, i] + h * acc
        x[0, i + 1] = x[0, i] + h * x[1, i + 1]
    return t, x


def implicit_euler2(f_implicit, t0, tn, x0, h):
    """Implicit Euler for 2nd-order ODEs.

    Requires a pre-solved expression for x_{k+1}.

    Parameters
    ----------
    f_implicit : callable(t_next, state_prev, h) -> float
        Returns the new acceleration from which position and velocity are
        updated.  Problem-specific closed-form.
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    for i in range(n):
        acc = f_implicit(t[i + 1], x[:, i], h)
        x[1, i + 1] = x[1, i] + h * acc
        x[0, i + 1] = x[0, i] + h * x[1, i + 1]
    return t, x


def midpoint2(f, t0, tn, x0, h):
    """Midpoint (leapfrog) method for 2nd-order ODEs.

    y_{i+1}  = y_i  + h · y'_i  + h²/2 · f(t_i, state_i)
    y'_{i+1} = y'_i + h/2 · (f(t_i, state_i) + f(t_{i+1}, state_{i+1}))

    2nd-order accurate; time-reversible.
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    for i in range(n):
        acc_i = f(t[i], x[:, i])
        x[0, i + 1] = x[0, i] + h * x[1, i] + 0.5 * h**2 * acc_i
        state_next = np.array([x[0, i + 1], x[1, i]])
        acc_next = f(t[i + 1], state_next)
        x[1, i + 1] = x[1, i] + 0.5 * h * (acc_i + acc_next)
    return t, x


def verlet(f, t0, tn, x0, h):
    """Standard Störmer-Verlet algorithm.

    Valid only when the acceleration does **not** depend on velocity (i.e.,
    conservative systems).  2nd-order accurate; time-reversible; symplectic.

    Uses explicit Euler for the first step.

    y_{i+1} = 2·y_i - y_{i-1} + h² · f(t_i, [y_i, 0])
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    # Bootstrap first step with Euler
    acc0 = f(t[0], x[:, 0])
    x[0, 1] = x[0, 0] + h * x[1, 0] + 0.5 * h**2 * acc0
    x[1, 1] = x[1, 0] + h * acc0
    for i in range(1, n):
        acc = f(t[i], x[:, i])
        x[0, i + 1] = 2 * x[0, i] - x[0, i - 1] + h**2 * acc
        x[1, i + 1] = (x[0, i + 1] - x[0, i - 1]) / (2 * h)
    return t, x


def velocity_verlet(f, t0, tn, x0, h):
    """Velocity Verlet (Leapfrog) algorithm.

    Valid only for conservative forces (acceleration independent of velocity).
    Better than plain Verlet in that velocity is available at each step and
    the algorithm needs only the current step — no look-back.

    y_{i+1}  = y_i + h·v_i + h²/2 · a_i
    v_{i+1}  = v_i + h/2  · (a_i + a_{i+1})
    """
    n, t = _make_grid(t0, tn, h)
    x = np.array([np.zeros(n + 1), np.zeros(n + 1)])
    x[:, 0] = x0
    acc = f(t[0], x[:, 0])
    for i in range(n):
        x[0, i + 1] = x[0, i] + h * x[1, i] + 0.5 * h**2 * acc
        acc_next = f(t[i + 1], np.array([x[0, i + 1], x[1, i]]))
        x[1, i + 1] = x[1, i] + 0.5 * h * (acc + acc_next)
        acc = acc_next
    return t, x
