# numphys — Numerical Methods for Physics

Personal library of clean, documented numerical methods assembled from
computational physics coursework and research practice.

## Contents

| Module | Methods |
|--------|---------|
| `ode.first_order` | Explicit Euler, Implicit Euler, Runge-Kutta 2/4, Adams 2 (explicit & implicit) |
| `ode.second_order` | Explicit Euler, Semi-implicit Euler, Implicit Euler, Midpoint, Störmer-Verlet, Velocity Verlet |
| `ode.systems` | Euler, RK2, RK4 for N-dimensional systems |
| `bvp.shooting` | Nonlinear shooting (bisection), Linear shooting |
| `bvp.finite_diff` | Thomas algorithm, Finite differences (linear BVP), Simple iterations (nonlinear BVP) |
| `pde.diffusion2d` | Explicit finite-difference solver for 2-D diffusion/reaction-diffusion |
| `integration.quadrature` | Trapezoidal rule, Simpson's rule, Midpoint rule |
| `stats.rng` | LCG uniform generator, Cauchy distribution sampler |
| `stats.analysis` | Histogram analysis, PDF comparison |
| `analysis.errors` | Max error, RMS error, Half-step (Richardson) error |
| `analysis.stability` | Stability plot overlay |
| `utils.plotting` | Solution plot, Phase portrait, 2-D diffusion animation |

## Quick start

```python
import numpy as np
from numphys.ode import runge_kutta4
from numphys.analysis import rms_error

# Solve  x' = -2x,  x(0)=1  on [0, 3]
f  = lambda t, x: -2 * x
sol = lambda t: np.exp(-2 * t)

t, x = runge_kutta4(f, t0=0, tn=3, x0=1.0, h=0.1)

h_values = np.logspace(-3, -1, 20)
h_arr, err_arr = rms_error(runge_kutta4, f, 0, 3, 1.0, sol, h_values)
print(f"RMS error at h=0.1: {err_arr[-1]:.2e}")
```

## Installation (editable)

```bash
pip install -e .
```

## Project structure

```
numphys/
├── ode/
│   ├── first_order.py     # 1st-order IVP solvers
│   ├── second_order.py    # 2nd-order IVP solvers (incl. Verlet)
│   └── systems.py         # N-dim ODE systems
├── bvp/
│   ├── shooting.py        # Shooting methods
│   └── finite_diff.py     # FD + Thomas algorithm
├── pde/
│   └── diffusion2d.py     # 2-D reaction-diffusion FD solver
├── integration/
│   └── quadrature.py      # Trapezoidal, Simpson, midpoint
├── stats/
│   ├── rng.py             # LCG generators
│   └── analysis.py        # Histogram / PDF tools
├── analysis/
│   ├── errors.py          # Error metrics vs step size
│   └── stability.py       # Stability overlay plots
└── utils/
    └── plotting.py        # Reusable plot helpers
```
