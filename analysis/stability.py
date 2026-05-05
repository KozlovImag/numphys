"""
Stability analysis helper
=========================
"""

import numpy as np
import matplotlib.pyplot as plt


def stability_plot(solver, f, t0, tn, x0, solution, h_values, ax=None, title="Stability analysis"):
    """Overlay numerical solutions for several step sizes against the analytical one.

    Parameters
    ----------
    solver   : callable(f, t0, tn, x0, h) -> (t, x)
    f        : ODE RHS
    t0, tn   : float
    x0       : initial condition
    solution : callable(t) -> analytical values (or None to skip overlay)
    h_values : iterable of float
    ax       : matplotlib Axes or None
    title    : str

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for h in h_values:
        t, x = solver(f, t0, tn, x0, h)
        y = np.asarray(x)
        row = y[0] if y.ndim > 1 else y
        ax.plot(t, row, label=f"h={h}")

    if solution is not None:
        t_fine = np.linspace(t0, tn, 500)
        sol = np.asarray(solution(t_fine))
        row = sol[0] if sol.ndim > 1 else sol
        ax.plot(t_fine, row, "k--", lw=2, label="Analytical")

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend(fontsize=8)
    ax.grid(True)
    return ax
