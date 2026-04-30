from .custom_optimizer import CustomOptimizer
from .line_search import LineSearchSolver, BacktrackingLineSearch, InterpolationLineSearch, BisectionLineSearch

from .curvature_estimator import (
    CurvatureEstimator,
    ExactBlockHessianCalculator,
    GaussNewtonBlockApproximation,
    NaiveIdentityCalculator,
    HutchinsonDiagonalApproximation,
)

from .numerical_optimizer import (
    NumericalOptimizer,
    LineSearchOptimizer,
)

from .modular_optimizer import (
    ModularNumericalOptimizer,
    ModularLineSearchOptimizer
)

from . import second_order
from .second_order import *

from . import first_order
from .first_order import *

from . import residual
from .residual import *

