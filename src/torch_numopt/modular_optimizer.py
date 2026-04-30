from __future__ import annotations
from .numerical_optimizer import NumericalOptimizer, LineSearchOptimizer


class ModularNumericalOptimizer(NumericalOptimizer):
    def __init__(self, model, scaling_matrix, lr_init=1, lr_method=None, solver="solve"):
        super().__init__(model, scaling_matrix, lr_init, lr_method, solver)


class ModularLineSearchOptimizer(LineSearchOptimizer):
    def __init__(self, model, scaling_matrix, line_search, lr_init=1, lr_method=None, solver="solve"):
        super().__init__(model, scaling_matrix, line_search, lr_init, lr_method, solver)
