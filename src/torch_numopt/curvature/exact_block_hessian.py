from __future__ import annotations
from typing import Optional
import logging
import torch
from functools import partial
from ..utils import param_dot, param_scalar_prod, param_add
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
            h_params = [None] * len(params)
            for i, _ in enumerate(params):
                h_params_block = torch.func.hessian(objective.loss, argnums=(i,))(*params)
                h_params[i] = self._reshape_hessian(h_params_block[0][0])

        else:
            # Calculate hessian for each batch and add the results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian matrix split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            h_params = None
            for i in range(objective.n_batches):
                # Calculate hessian of the batch
                batched_loss = partial(objective.loss, batch_idx=i)

                h_param_batch = list(torch.func.hessian(batched_loss, argnums=tuple(range(len(params))))(*params))
                for j, _ in enumerate(h_param_batch):
                    h_param_batch[j] = self._reshape_hessian(h_param_batch[j][j])

                if objective.reduction == "mean":
                    h_param_batch = param_scalar_prod(objective.batch_data_size(i), h_param_batch)

                # Aggregate result
                if h_params is None:
                    h_params = h_param_batch
                else:
                    h_params = param_add(h_params, h_param_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the exact hessian...", i)

            if objective.reduction == "mean":
                h_params = param_scalar_prod(1 / objective.data_size, h_params)

        h_params = list(h_params)

        # Damp matrix
        if self.damping is not None:
            for i, h in enumerate(h_params):
                if self.damping == "identity":
                    h_params[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_params[i] = h + self.mu * torch.diag(h.diagonal())
                else:
                    raise ValueError(f"Invalid damping strategy {self.damping}.")

        return tuple(h_params)

    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        if not objective.batched:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian vector product.")

            zero_vector = tuple(torch.zeros_like(p) for p in params)
            hess_dot_step = [None] * len(params)
            for i, s_d in enumerate(step_dir):
                block_step_dir = zero_vector[:i] + (s_d,) + zero_vector[i + 1 :]
                _, hess_dot_step_block = torch.autograd.functional.vhp(objective.loss, params, v=block_step_dir)
                hess_dot_step[i] = hess_dot_step_block[i]
            hess_dot_step = tuple(hess_dot_step)

            # loss = objective.loss(*params)
            # grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            # hess_dot_step = [None] * len(params)
            # for i, (p, s, g) in enumerate(zip(params, step_dir, grad)):
            #     if torch.all(s == 0):
            #         hess_dot_step.append(torch.zeros_like(p))
            #         continue

            #     grad_dot_s = param_dot(g, s)
            #     hvp_i = torch.autograd.grad(grad_dot_s, p, retain_graph=True, create_graph=False)[0]
            #     hess_dot_step[i] = hvp_i

            # hess_dot_step = tuple(hess_dot_step)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the exact hessian vector product split in %d batches of size %d.", objective.n_batches, objective.batch_size)

            hess_dot_step = None
            for i in range(objective.n_batches):
                # Calculate hessian of the batch

                batched_loss = objective.loss(*params, batch_idx=i)
                hess_dot_step_batch = [None] * len(params)
                for i, (p, s_d) in enumerate(zip(params, step_dir)):
                    grad_fn = torch.func.grad(batched_loss, argnums=i)
                    tangents = zero_params[:i] + (s_d,) + zero_params[i + 1 :]
                    _, hess_dot_step_p = torch.func.jvp(grad_fn, params, tuple(tangents))
                    hess_dot_step_batch[i] = hess_dot_step_p
                hess_dot_step_batch = tuple(hess_dot_step_batch)

                if objective.reduction == "mean":
                    hess_dot_step_batch = param_scalar_prod(objective.batch_data_size(i), hess_dot_step_batch)

                if hess_dot_step is None:
                    hess_dot_step = hess_dot_step_batch
                else:
                    hess_dot_step = param_add(hess_dot_step, hess_dot_step_batch)

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Computed batch %d for the exact hessian vector product...", i)

            if objective.reduction == "mean":
                hess_dot_step = param_scalar_prod(1 / objective.data_size, hess_dot_step)

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
