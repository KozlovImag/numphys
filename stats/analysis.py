"""
Statistical analysis helpers
============================
"""

import numpy as np
import matplotlib.pyplot as plt


def histogram_analysis(samples, n_bins=50, density=True, ax=None, label=None):
    """Plot a normalised histogram of ``samples``.

    Parameters
    ----------
    samples  : array-like
    n_bins   : int
    density  : bool — normalise to probability density if True
    ax       : matplotlib Axes or None
    label    : str or None

    Returns
    -------
    counts, bin_edges
    """
    if ax is None:
        _, ax = plt.subplots()
    counts, edges, _ = ax.hist(samples, bins=n_bins, density=density, alpha=0.7, label=label)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True)
    if label:
        ax.legend()
    return counts, edges


def compare_pdf(samples, pdf, x_range=None, n_bins=50, ax=None):
    """Overlay a histogram with a theoretical PDF.

    Parameters
    ----------
    samples : array-like
    pdf     : callable(x) -> float — theoretical probability density
    x_range : (float, float) or None — defaults to sample min/max
    n_bins  : int
    ax      : matplotlib Axes or None
    """
    if ax is None:
        _, ax = plt.subplots()
    histogram_analysis(samples, n_bins=n_bins, density=True, ax=ax, label="Empirical")
    if x_range is None:
        lo, hi = np.percentile(samples, [1, 99])
    else:
        lo, hi = x_range
    xs = np.linspace(lo, hi, 500)
    ax.plot(xs, [pdf(x) for x in xs], "r-", lw=2, label="Theoretical PDF")
    ax.legend()
    return ax
