"""
Shooting methods for boundary value problems
============================================
Solve the BVP

    y''(t) = f(t, y, y'),   y(t0) = y0,  y(tn) = yn

by converting it to an IVP and scanning over the initial slope y'(t0).
"""

import numpy as np
from ..ode.second_order import midpoint2


def shooting(f, t0, tn, y0, yn, h, v_range=(-50, 50), tol=1e-4, max_iter=1000, ivp_solver=None):
    """Nonlinear shooting method — bisection on the initial slope.

    Parameters
    ----------
    f         : callable(t, state) -> float  — 2nd-order ODE RHS
    t0, tn    : float  — interval endpoints
    y0        : float  — left boundary condition  y(t0) = y0
    yn        : float  — right boundary condition y(tn) = yn
    h         : float  — integration step
    v_range   : (float, float)  — search interval for initial slope y'(t0)
    tol       : float  — bisection convergence tolerance
    max_iter  : int    — maximum bisection iterations
    ivp_solver: callable  — IVP solver with signature solver(f, t0, tn, x0, h)
                            (default: midpoint2)

    Returns
    -------
    t  : np.ndarray
    x  : np.ndarray, shape (2, n+1)
    v_opt : float  — optimal initial slope found
    """
    if ivp_solver is None:
        ivp_solver = midpoint2

    va, vb = v_range

    def shoot(v):
        t, x = ivp_solver(f, t0, tn, np.array([y0, v]), h)
        return x[0, -1] - yn

    for _ in range(max_iter):
        vc = (va + vb) / 2
        fa, fc = shoot(va), shoot(vc)
        if abs(fc) < tol:
            break
        if fa * fc < 0:
            vb = vc
        else:
            va = vc

    t, x = ivp_solver(f, t0, tn, np.array([y0, vc]), h)
    return t, x, vc


def shooting_linear(f, t0, tn, y0, yn, h, ivp_solver=None):
    """Linear shooting method for **linear** BVPs.

    Decomposes the solution as  y = u + λ·v  where u and v satisfy
    auxiliary IVPs.  Only two forward integrations needed (no iteration).

    Applicable only when f is linear in y and y'.

    Parameters
    ----------
    f         : callable(t, state) -> float
    t0, tn    : float
    y0        : float  — y(t0)
    yn        : float  — y(tn)
    h         : float
    ivp_solver: callable  — default midpoint2

    Returns
    -------
    t : np.ndarray
    x : np.ndarray, shape (2, n+1)
    lam : float  — mixing coefficient
    """
    if ivp_solver is None:
        ivp_solver = midpoint2

    # Solve u'' = f(t,u,u'),  u(t0)=y0, u'(t0)=0
    t, xu = ivp_solver(f, t0, tn, np.array([y0, 0.0]), h)
    # Solve v'' = f(t,v,v'),  v(t0)=0,  v'(t0)=1
    _, xv = ivp_solver(f, t0, tn, np.array([0.0, 1.0]), h)

    lam = (yn - xu[0, -1]) / xv[0, -1]
    x = xu + lam * xv
    return t, x, lam
