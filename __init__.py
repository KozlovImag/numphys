"""
numphys — Numerical Methods for Physics
========================================
A personal library of numerical methods for solving ODEs, PDEs,
boundary value problems, and statistical modelling tasks.

Modules
-------
ode         : ODE solvers (Euler, Runge-Kutta, Adams, Verlet, ...)
bvp         : Boundary value problem solvers (shooting, finite differences)
pde         : PDE solvers (2-D finite difference diffusion)
integration : Quadrature rules (trapezoidal, ...)
stats       : Random-number generators and distribution utilities
analysis    : Error / stability analysis helpers
utils       : Plotting helpers
"""

__version__ = "0.1.0"
__author__ = "Oleksii Kozlov"

from . import ode, bvp, pde, integration, stats, analysis, utils
