"""
torch_numopt: Second-order optimization algorithms for PyTorch.

This package provides a collection of numerical optimizers (Newton, Gauss-Newton,
Levenberg-Marquardt, conjugate gradient, L-BFGS, AdaHessian, etc.) that utilize
exact or approximate curvature information. It also includes line-search and
trust-region frameworks, curvature estimators (Hessian, Gauss-Newton, diagonal
approximations), and a variety of linear solvers.
"""

from .objective import ObjectiveFunction, SupervisedLearningObjective
from .curvature_estimator import CurvatureEstimator
from .line_search import LineSearchSolver, BacktrackingLineSearch, InterpolationLineSearch, BisectionLineSearch, create_line_search_solver
from .trust_region import TrustRegionSolver, CauchyPointTRSolver, DoglegTRSolver, create_trust_region_solver
from .numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer

from . import curvature
from .curvature import *

from . import algorithms
from .algorithms import *

from . import utils
from .utils import *

__all__ = [
    "ObjectiveFunction",
    "SupervisedLearningObjective",
    "CurvatureEstimator",
    "NumericalOptimizer",
    "LineSearchOptimizer",
    "TrustRegionOptimizer",
    "LineSearchSolver",
    "BacktrackingLineSearch",
    "InterpolationLineSearch",
    "BisectionLineSearch",
    "TrustRegionSolver",
    "CauchyPointTRSolver",
    "DoglegTRSolver",
    "create_line_search_solver",
    "create_trust_region_solver",
    *algorithms.__all__,
    *curvature.__all__,
    *utils.__all__,
]

__version__ = "1.0.0"
