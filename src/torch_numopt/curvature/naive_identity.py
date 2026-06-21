from __future__ import annotations
from typing import Iterable
import torch
from ..curvature_estimator import CurvatureEstimator
from ..utils import param_dot


class NaiveIdentityCalculator(CurvatureEstimator):
    """
    Naive second derivative approximation. Always assumes an identity as the hessian.
    """

    def __init__(self):
        super().__init__(ndim=0, uses_blocks=False)

    def scaling_matrix(self, objective, params) -> float:
        return 1

    def hvp(self, objective, params, step_dir) -> Iterable[torch.Tensor]:
        return step_dir

    def quadratic_form(self, objective, params, step_dir) -> torch.Tensor:
        return param_dot(step_dir, step_dir)
