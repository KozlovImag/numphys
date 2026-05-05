"""Statistical utilities and random-number generators."""
from .rng import lcg_uniform, cauchy_rng
from .analysis import histogram_analysis, compare_pdf
__all__ = ["lcg_uniform", "cauchy_rng", "histogram_analysis", "compare_pdf"]
