"""
Concrete optimizer implementations.

This package contains ready-to-use optimizers that inherit from the base classes
in :mod:`numerical_optimizer`. They combine curvature estimators with step-
selection strategies (line search or trust region) to provide complete
optimization algorithms.

Available optimizers:
- GradientDescent (and variants with line search / trust region)
- ConjugateGradient (with line search)
- Newton (exact Hessian, with line search, trust region, or CG)
- GaussNewton (Gauss-Newton approximation)
- LevenbergMarquardt (trust-region with adaptive damping)
- LBFGS (limited-memory BFGS)
- AdaHessian (diagonal Hessian with momentum)
"""

from .gradient_descent import GradientDescent, GradientDescentLS, GradientDescentTR, GradientDescentLipschitz
from .conjugate_gradient import ConjugateGradient, ConjugateGradientLS
from .newton import Newton, NewtonLS, NewtonTR, NewtonCG, NewtonCGLS, NewtonCGTR
from .gauss_newton import GaussNewton, GaussNewtonLS, GaussNewtonTR
from .levenberg_marquardt import LevenbergMarquardt
from .lbfgs import LBFGS, LBFGSLS
from .adahessian import AdaHessian, AdaHessianLS

__all__ = [
    "GradientDescent",
    "GradientDescentLS",
    "GradientDescentTR",
    "GradientDescentLipschitz",
    "ConjugateGradient",
    "ConjugateGradientLS",
    "Newton",
    "NewtonLS",
    "NewtonTR",
    "NewtonCG",
    "NewtonCGLS",
    "NewtonCGTR",
    "GaussNewton",
    "GaussNewtonLS",
    "GaussNewtonTR",
    "LevenbergMarquardt",
    "LBFGS",
    "LBFGSLS",
    "AdaHessian",
    "AdaHessianLS",
]
