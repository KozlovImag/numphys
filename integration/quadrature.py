"""
Numerical quadrature rules
===========================
"""

import math
import numpy as np


def trapezoidal(f, a, b, h):
    """Composite trapezoidal rule.

    ∫_a^b f(x) dx ≈ h · [f(a)/2 + f(x_1) + ... + f(x_{n-1}) + f(b)/2]

    2nd-order accuracy in h.

    Parameters
    ----------
    f : callable(x) -> float
    a, b : float  — integration limits
    h    : float  — step size

    Returns
    -------
    float — approximate integral value
    """
    n = round((b - a) / h)
    x = np.array([a + i * h for i in range(n + 1)])
    w = np.ones(n + 1)
    w[0] = w[-1] = 0.5
    return h * np.sum(w * np.array([f(xi) for xi in x]))


def simpsons(f, a, b, n=None, h=None):
    """Composite Simpson's 1/3 rule.

    4th-order accuracy.  n must be even; if not supplied, derived from h.

    Parameters
    ----------
    f    : callable(x) -> float
    a, b : float
    n    : int (even)  — number of sub-intervals (takes precedence over h)
    h    : float       — step size (used if n is None)

    Returns
    -------
    float
    """
    if n is None and h is None:
        raise ValueError("Provide either n or h.")
    if n is None:
        n = math.ceil((b - a) / h)
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    return (x[1] - x[0]) / 3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])


def midpoint_rule(f, a, b, h):
    """Composite midpoint (rectangle) rule.

    2nd-order accuracy.

    Parameters
    ----------
    f    : callable(x) -> float
    a, b : float
    h    : float

    Returns
    -------
    float
    """
    n = round((b - a) / h)
    x_mid = np.array([a + (i + 0.5) * h for i in range(n)])
    return h * np.sum([f(xi) for xi in x_mid])
