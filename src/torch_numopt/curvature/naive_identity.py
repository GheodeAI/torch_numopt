from __future__ import annotations
from typing import Iterable
import torch
from ..curvature_estimator import CurvatureEstimator


class NaiveIdentityCalculator(CurvatureEstimator):
    """
    Naive second derivative approximator. Always assumes an identity as the hessian.
    """

    def scaling_matrix(self) -> None:
        return None

    def hvp(self, step_dir) -> Iterable[torch.Tensor]:
        return step_dir

    def quadratic_form(self, step_dir) -> torch.Tensor:
        return sum(torch.sum(s_i * s_i) for s_i in step_dir)
