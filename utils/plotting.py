"""
Plotting utilities
==================
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_solution(t, x, label="Numerical", solution=None, title="Solution", ax=None):
    """Plot a 1-D (or 2nd-order) ODE solution.

    Parameters
    ----------
    t        : np.ndarray — time grid
    x        : np.ndarray — shape (n+1,) or (2, n+1)
    label    : str
    solution : callable(t) -> values, or None
    title    : str
    ax       : matplotlib Axes or None

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    x = np.asarray(x)
    row = x[0] if x.ndim > 1 else x
    ax.plot(t, row, label=label)
    if solution is not None:
        sol = np.asarray(solution(t))
        sol_row = sol[0] if sol.ndim > 1 else sol
        ax.plot(t, sol_row, "k--", label="Analytical")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend()
    ax.grid(True)
    return ax


def plot_phase_portrait(x, labels=("x₁", "x₂"), title="Phase portrait", ax=None):
    """Phase portrait for 2-D systems.

    Parameters
    ----------
    x      : np.ndarray, shape (N, n+1) — N variables over time
    labels : tuple of str
    title  : str
    ax     : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x = np.asarray(x)
    ax.plot(x[0], x[1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.grid(True)
    return ax


def animate_diffusion2d(T, X, Y, U, interval=50, step=1):
    """Create a matplotlib animation of a 2-D diffusion solution.

    Parameters
    ----------
    T, X, Y : arrays from ``diffusion2d``
    U       : list of 2-D snapshots
    interval: int — milliseconds between frames
    step    : int — use every n-th snapshot

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation

    U_sel = U[::step]
    T_sel = T[::step]
    z_min = np.min([u.min() for u in U_sel])
    z_max = np.max([u.max() for u in U_sel])
    N = 100
    levels = np.linspace(z_min, z_max, N + 1)

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        ax.clear()
        ax.set_title(f"t = {T_sel[i]:.2f}")
        ax.contourf(X, Y, U_sel[i].T, levels=levels, cmap="hot_r")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    anim = FuncAnimation(fig, update, frames=len(U_sel), interval=interval)
    return anim
