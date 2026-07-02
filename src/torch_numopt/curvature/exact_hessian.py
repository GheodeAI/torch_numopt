"""
Exact Hessian computation using ``torch.func``.

This module provides the full Hessian matrix (or block-diagonal approximation)
via automatic differentiation. It supports damping (identity or Fletcher) and
batched evaluation.
"""

from __future__ import annotations
from typing import Optional
import logging
import torch
from functools import partial
from ..utils import param_dot, param_scalar_prod, param_add, param_argnums, param_numel
from ..curvature_estimator import CurvatureEstimator
from ..objective import ObjectiveFunction
from ..utils import Params

logger = logging.getLogger(__name__)


class ExactHessianCalculator(CurvatureEstimator):
    """
    Compute the exact Hessian matrix (full or block) of the objective.

    The Hessian is obtained using ``torch.func.hessian``, which computes the
    full second-order derivatives. For large models, this can be memory-
    intensive; use the block version for parameter groups.

    Parameters
    ----------
    damping : str or None, default=None
        Damping strategy: ``"identity"`` adds ``mu * I``, ``"fletcher"`` adds
        ``mu * diag(H)``. If ``None``, no damping is applied.
    mu : float, default=1e-4
        Damping coefficient.
    """

    def __init__(
        self,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(ndim=2, uses_blocks=False)
        self.damping = damping
        self.mu = mu

    def _construct_hessian(self, h_params: Params, params) -> torch.Tensor:
        n_groups = len(h_params)
        row_blocks = []
        for i in range(n_groups):
            col_blocks = []
            for j in range(n_groups):
                block = h_params[i][j].reshape(params[i].numel(), params[j].numel())
                col_blocks.append(block)
            row_blocks.append(torch.cat(col_blocks, dim=1))
        full_hessian = torch.cat(row_blocks, dim=0)
        return full_hessian

    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Params:
        """
        Calculation of the exact hessian of the Neural network given a dataset.

        Parameters
        ----------
        x: torch.Tensor
            Input dataset for calculating the loss.
        y: torch.Tensor
            Target dataset for calculating the loss.
        loss_fn: torch.Module
            Loss function for which to calculate the hessian.
        vectorize: boolean
            Use vectorization in pytorch's implementation of the hessian calculation.
        """

        # Calculate exact Hessian matrix
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian matrix.")

            # Calculate hessian with every sample in the dataset
            h_params = torch.func.hessian(objective.loss, argnums=tuple(range(len(params))))(*params)
            h_params = self._reshape_hessian(self._construct_hessian(h_params, params))

        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian matrix split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            h_params = None
            for i in range(objective.n_batches):
                # Calculate hessian of the batch
                batched_loss = partial(objective.loss, batch_idx=i)

                h_param_batch = torch.func.hessian(batched_loss, argnums=tuple(range(len(params))))(*params)
                h_param_batch = self._reshape_hessian(self._construct_hessian(h_param_batch, params))

                if objective.reduction == "mean":
                    h_param_batch = objective.batch_data_size(i) * h_param_batch

                # Aggregate result
                if h_params is None:
                    h_params = h_param_batch
                else:
                    h_params = h_params + h_param_batch

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the exact hessian...", i)

            if objective.reduction == "mean":
                h_params = h_params / objective.data_size

        # Damp matrix
        if self.damping is not None:
            if self.damping == "identity":
                h_params.diagonal().add_(self.mu)
            elif self.damping == "fletcher":
                # n = param_numel(params)
                # h_params[torch.arange(n), torch.arange(n)] += h_params.diagonal()
                h_params.diagonal().add_(h_params.diagonal())
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return h_params

    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.info("Computing the exact hessian vector product.")
            _, hess_dot_step = torch.autograd.functional.vhp(objective.loss, params, v=step_dir)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian vector product split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            hess_dot_step = None
            for i in range(objective.n_batches):
                batched_loss = partial(objective.loss, batch_idx=i)
                _, hess_dot_step_batch = torch.autograd.functional.vhp(batched_loss, params, v=step_dir)

                if objective.reduction == "mean":
                    hess_dot_step_batch = objective.batch_data_size(i) * hess_dot_step_batch

                if hess_dot_step is None:
                    hess_dot_step = hess_dot_step_batch
                else:
                    hess_dot_step = param_add(hess_dot_step, hess_dot_step_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the exact hessian vector product...", i)

            if objective.reduction == "mean":
                hess_dot_step = hess_dot_step / objective.data_size

        # Damp vector
        if self.damping is not None:
            if self.damping == "identity":
                hess_dot_step = param_add(hess_dot_step, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return hess_dot_step

    def quadratic_form(self, objective, params: Params, grad_params: Params) -> torch.Tensor:
        return param_dot(grad_params, self.hvp(objective, params, grad_params))
