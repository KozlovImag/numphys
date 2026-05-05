from setuptools import setup, find_packages

setup(
    name="numphys",
    version="0.1.0",
    author="Oleksii Kozlov",
    description="Numerical methods for physics: ODE/PDE solvers, BVP, quadrature, statistics.",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy", "matplotlib"],
    extras_require={"dev": ["scipy", "pytest", "jupyter"]},
)
