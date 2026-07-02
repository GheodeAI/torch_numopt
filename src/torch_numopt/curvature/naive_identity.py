"""
Identity curvature estimator (no curvature).

This estimator treats the Hessian as the identity matrix, i.e., it ignores
second-order information and effectively performs gradient descent.
"""

from __future__ import annotations
from typing import Iterable
import torch
from ..curvature_estimator import CurvatureEstimator
from ..utils import param_dot


class NaiveIdentityCalculator(CurvatureEstimator):
    """
    Curvature estimator that always returns the identity matrix.

    The scaling matrix is 1 (scalar), the Hessian-vector product is the vector
    itself, and the quadratic form is the squared norm.
    """

    def __init__(self):
        super().__init__(ndim=0, uses_blocks=False)

    def scaling_matrix(self, objective, params) -> float:
        return torch.tensor(1, device=params[0].device, dtype=params[0].dtype)

    def hvp(self, objective, params, step_dir) -> Iterable[torch.Tensor]:
        return step_dir

    def quadratic_form(self, objective, params, step_dir) -> torch.Tensor:
        return param_dot(step_dir, step_dir)
