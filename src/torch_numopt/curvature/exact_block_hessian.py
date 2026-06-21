from __future__ import annotations
from typing import Iterable, Optional
import logging
from copy import copy
import torch
from torch import nn
from functools import partial
from ..utils import param_dot, param_scalar_prod, param_add, param_sizes
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator
from ..objective import ObjectiveFunction
from ..utils import Params

logger = logging.getLogger(__name__)


class ExactBlockHessianCalculator(CurvatureEstimator):
    """
    Approximates the hessian in blocks, only taking the inner-layer second derivatives.
    """

    def __init__(
        self,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(ndim=2, uses_blocks=True)
        self.damping = damping
        self.mu = mu

    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Iterable:
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
            logger.info("Computing the exact hessian matrix.")

            # Calculate hessian with every sample in the dataset
            h_params = list(torch.func.hessian(objective.loss, argnums=tuple(range(len(params))))(*params))
            for i, _ in enumerate(h_params):
                h_params[i] = self._reshape_hessian(h_params[i][i])

        else:
            # Calculate hessian for each batch and add the results
            logger.info("Computing the exact hessian matrix split in %d batches of size %d.", len(objective.n_batches), objective.batch_size)

            h_params = []
            for i in range(objective.n_batches):
                # Calculate hessian of the batch
                batched_loss = partial(objective.loss, batch_idx = i)

                h_param_batch = list(torch.func.hessian(batched_loss, argnums=tuple(range(len(params))))(*params))
                for j, _ in enumerate(h_param_batch):
                    h_param_batch[j] = self._reshape_hessian(h_param_batch[j][j])

                # Aggregate result
                if h_params == []:
                    h_params = h_param_batch
                else:
                    h_params = param_add(h_params, h_param_batch)

                logger.info("Computed batch %d for the exact hessian...", i)

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the exact hessian...")
            for i, h in enumerate(h_params):
                if self.damping == "identity":
                    h_params[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_params[i] = h + self.mu * h.diagonal()
                else:
                    raise ValueError(f"Invalid damping strategy {self.damping}.")

        return h_params

    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        logger.info("Computing the product p^T H p.")

        if not objective.batched:
            _, hess_dot_step = torch.autograd.functional.hvp(objective.loss, tuple(params), v=tuple(step_dir))
        else:
            logger.info("Computing the exact hessian vector product split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            hess_dot_step = None
            for i in range(objective.n_batches):
                # Calculate hessian of the batch
                batched_loss = partial(objective.loss, batch_idx=i)

                _, hess_dot_step_batch = torch.autograd.functional.hvp(batched_loss, tuple(params), v=tuple(step_dir))

                if hess_dot_step is None:
                    hess_dot_step = hess_dot_step_batch
                else:
                    hess_dot_step = param_add(hess_dot_step, hess_dot_step_batch)
                logger.info("Computed batch %d for the exact hessian vector product...", i)

        # Damp vector
        if self.damping is not None:
            logger.info("Applying damping to the exact hessian...")
            if self.damping == "identity":
                hess_dot_step = param_add(hess_dot_step, param_scalar_prod(self.mu, step_dir))
            elif self.damping == "fletcher":
                raise NotImplementedError("Fletcher damping not available for hvp.")
            else:
                raise ValueError(f"Invalid damping strategy {self.damping}.")

        return hess_dot_step

    def quadratic_form(self, objective, params: Params, d_p_list: Params) -> torch.Tensor:
        return param_dot(d_p_list, self.hvp(objective, params, d_p_list))
