""" """

from __future__ import annotations
from typing import Iterable, Callable
from abc import ABC, abstractmethod
import logging
import torch
from torch import nn
from functools import reduce, partial
from .utils import param_reshape_like
from torch.func import functional_call
from copy import copy

logger = logging.getLogger(__name__)

class CurvatureEstimator(ABC):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int | None = None,
    ):
        self.model = model
        self.param_keys = dict(model.named_parameters()).keys()
        self.params = tuple(model.parameters())
        self.batch_size = batch_size

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
    
    def store_data(self, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable):
        """
        Stores the necessary data for later use
        """

        self.x_ = x
        self.y_ = y
        self.loss_fn_ = loss_fn
    
    def reset(self):
        """
        Resets the parameters of the curvature estimator. 

        Used primarily for quasi-newton methods with memory.
        """

    @abstractmethod
    def scaling_matrix(self) -> Iterable | None:
        """
        Obtains the second derivative approximation.
        """
    
    @abstractmethod
    def hvp(self, step_dir: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        """
        Compute B_k p^T
        with B being the scaling matrix and p the step direction
        """

    @abstractmethod
    def quadratic_form(self, step_dir: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute p B_k p^T
        with B being the scaling matrix and p the step direction
        """





