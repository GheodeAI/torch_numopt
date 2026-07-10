"""Base class for curvature estimation class."""

from __future__ import annotations
from typing import Iterable
from abc import ABC, abstractmethod
import logging
import torch
from functools import reduce
from .objective import ObjectiveFunction
from .utils import Params, param_numel, param_flatten, param_sizes

logger = logging.getLogger(__name__)


class CurvatureEstimator(ABC):
    """
    Abstract base class for all curvature estimators.

    A curvature estimator provides a matrix (or matrix-vector product) that
    approximates the Hessian of the objective. This can be the exact Hessian,
    Gauss-Newton, diagonal, or identity.

    Subclasses must implement:
    - ``scaling_matrix``: return the matrix (or block/scalar representation).
    - ``hvp``: Hessian-vector product.
    - ``quadratic_form``: pᵀ H p.
    """

    def __init__(self, ndim: int = 2, uses_blocks: bool = False):
        self.ndim = ndim
        self.uses_blocks = uses_blocks

    @staticmethod
    def _reshape_hessian(hess: torch.Tensor):
        """
        Reshape a misshapen Hessian tensor into a square matrix.

        The input is expected to have an even number of dimensions. The first
        half of dimensions are combined into the rows, the second half into the
        columns.

        Parameters
        ----------
        hess : torch.Tensor
            Hessian tensor from ``torch.func.hessian``.

        Returns
        -------
        torch.Tensor
            Square matrix.
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
        Reset any internal state (intended to be used for quasi-Newton methods).
        By default does nothing.
        """

    def full_scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> torch.Tensor:
        """
        Return the curvature matrix as a single dense tensor.

        Depending on the estimator's ``ndim`` and ``uses_blocks``, this method
        constructs a full matrix from the internal representation.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Parameter tensors.

        Returns
        -------
        torch.Tensor
            Full square matrix of size (total_params, total_params).
        """

        n_params = param_numel(params)
        B = self.scaling_matrix(objective, params)
        if self.ndim == 0:
            full_B = torch.zeros((n_params, n_params), device=B.device, dtype=B.dtype)
            full_B.fill_diagonal_(B)
        if self.ndim == 1:
            if self.uses_blocks:
                B = param_flatten(B)
            full_B = torch.zeros((n_params, n_params), device=B.device, dtype=B.dtype)
            full_B.diagonal().copy_(B)
        elif self.ndim == 2:
            if self.uses_blocks:
                full_B = torch.block_diag(*B)
            else:
                full_B = B
        return full_B

    @abstractmethod
    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Iterable | torch.Tensor:
        """
        Obtain the curvature matrix in its native representation.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Parameter tensors.

        Returns
        -------
        iterable or torch.Tensor
            Representation of the matrix (scalar, vector, tuple of blocks, or
            full tensor).
        """

    @abstractmethod
    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        """
        Compute the Hessian-vector product H * v.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Parameter tensors.
        step_dir : Params
            Vector v (same structure as params).

        Returns
        -------
        Params
            Result of H * v.
        """

    @abstractmethod
    def quadratic_form(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> torch.Tensor:
        """
        Compute the quadratic form vᵀ H v.

        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function.
        params : Params
            Parameter tensors.
        step_dir : Params
            Vector v.

        Returns
        -------
        torch.Tensor
            Scalar value vᵀ H v.
        """
