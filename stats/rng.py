"""
Random number generators
========================
Pure-Python LCG-based generators as implemented in the lab notebooks.
These are intentionally simple educational implementations — not
suitable for security or high-quality Monte-Carlo work (use
``numpy.random`` for that).
"""

import time
import math
import numpy as np


def lcg_uniform(n, a=0.0, b=1.0, *, m=17, c=5, z=2**10, seed=None):
    """Linear Congruential Generator (LCG) — uniform distribution on [a, b].

    x_{i+1} = (m · x_i + c)  mod  z
    r_i      = x_i / z          ∈ [0, 1)
    sample_i = a + (b - a) · r_i

    Parameters
    ----------
    n    : int   — number of samples
    a, b : float — output range
    m, c : int   — multiplier and increment
    z    : int   — modulus
    seed : float or None — initial state (uses perf_counter if None)

    Returns
    -------
    np.ndarray of shape (n,)
    """
    x = time.perf_counter() if seed is None else float(seed)
    samples = []
    for _ in range(n):
        x = (m * x + c) % z
        r = x / z
        samples.append(a + (b - a) * r)
    return np.array(samples)


def cauchy_rng(n, x0=0.0, gamma=1.0, *, m=17, c=5, z=2**10, seed=None):
    """LCG-based Cauchy (Lorentz) distribution sampler.

    Uses the inverse CDF:  X = x0 + γ · tan(π · U)  where U ~ Uniform(0,1).

    Parameters
    ----------
    n     : int   — number of samples
    x0    : float — location parameter (median)
    gamma : float — scale parameter (half-width at half-maximum)
    m, c, z, seed : see ``lcg_uniform``

    Returns
    -------
    np.ndarray of shape (n,)
    """
    u = lcg_uniform(n, 0.0, 1.0, m=m, c=c, z=z, seed=seed)
    return x0 + gamma * np.tan(math.pi * u)
