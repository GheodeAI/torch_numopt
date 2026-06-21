""" """

from __future__ import annotations
from typing import Iterable
from abc import ABC, abstractmethod
import logging
import torch
from functools import reduce
from .objective import ObjectiveFunction
from .utils import Params

logger = logging.getLogger(__name__)


class CurvatureEstimator(ABC):
    def __init__(self, ndim: int = 2, uses_blocks: bool = False):
        self.ndim = ndim
        self.uses_blocks = uses_blocks

    @staticmethod
    def _reshape_hessian(hess: torch.Tensor):
        """
        Procedure to reshape a misshapen hessian matrix.
        The input is expected to be an array of size :math:`(X,Y,...,X,Y,...)` and the output will be
        a square matrix of size :math:`(X \cdot Y \cdots, X \cdot Y \cdots)`.


        Parameters
        ----------

        hess: torch.Tensor
            Misshapen hessian matrix.
        """

        if hess.dim() == 2:
            return hess

        if hess.dim() % 2 != 0:
            raise ValueError("Hessian has an incorrect shape.")

        # Divide shape in two halves, multiply each half to get total size
        new_shape = (
            reduce(lambda x, y: x * y, hess.size()[hess.dim() // 2 :]),
            reduce(lambda x, y: x * y, hess.size()[: hess.dim() // 2]),
        )

        assert new_shape[0] == new_shape[1], "Hessian an the incorrect shape."

        return hess.reshape(new_shape)

    def reset(self):
        """
        Resets the parameters of the curvature estimator.

        Used primarily for quasi-newton methods with memory.
        """

    @abstractmethod
    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Iterable | None:
        """
        Obtains the second derivative approximation.
        """

    @abstractmethod
    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        """
        Compute B_k p^T
        with B being the scaling matrix and p the step direction
        """

    @abstractmethod
    def quadratic_form(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> torch.Tensor:
        """
        Compute p B_k p^T
        with B being the scaling matrix and p the step direction
        """
