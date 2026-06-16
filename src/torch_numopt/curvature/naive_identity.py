from __future__ import annotations
from typing import Iterable
import torch
from torch import nn
from ..curvature_estimator import CurvatureEstimator


class NaiveIdentityCalculator(CurvatureEstimator):
    """
    Naive second derivative approximator. Always assumes an identity as the hessian.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model=model, ndim=0, uses_blocks=False)

    def scaling_matrix(self) -> float:
        return 1

    def hvp(self, step_dir) -> Iterable[torch.Tensor]:
        return step_dir

    def quadratic_form(self, step_dir) -> torch.Tensor:
        return sum(torch.sum(s_i * s_i) for s_i in step_dir)
