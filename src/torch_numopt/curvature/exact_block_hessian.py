from __future__ import annotations
from typing import Iterable, Optional
import logging
from copy import copy
import torch
from torch import nn
from functools import reduce, partial
from ..utils import param_reshape_like
from torch.func import functional_call
from ..curvature_estimator import CurvatureEstimator

logger = logging.getLogger(__name__)


class ExactBlockHessianCalculator(CurvatureEstimator):
    """
    Approximates the hessian in blocks, only taking the inner-layer second derivatives.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: Optional[int] = None,
        damping: Optional[str] = None,
        mu: float = 1e-4,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.damping = damping
        self.mu = mu

    def scaling_matrix(self) -> Iterable:
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

        loss_fn = copy(self.loss_fn_)
        is_mean = loss_fn.reduction == "mean"
        if is_mean:
            loss_fn.reduction = "sum"

        scale = 1 / len(self.x_) if is_mean else 1

        def eval_model_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the exact hessian matrix.")

            # Calculate hessian with every sample in the dataset
            eval_model = partial(eval_model_batch, x=self.x_, y=self.y_)

            h_list = list(torch.func.hessian(eval_model, argnums=tuple(range(len(self.params))))(*self.params))
            for i, _ in enumerate(h_list):
                h_list[i] = self._reshape_hessian(h_list[i][i] * scale)

        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the exact hessian matrix split in %d batches of size %d.", len(batch_start), self.batch_size)

            h_list = []
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate hessian of the batch
                eval_model = partial(eval_model_batch, x=x_batch, y=y_batch)

                h_list_batch = list(torch.func.hessian(eval_model, argnums=tuple(range(len(self.params))))(*self.params))
                for j, _ in enumerate(h_list_batch):
                    h_list_batch[j] = self._reshape_hessian(h_list_batch[j][j]) * scale

                # Aggregate result
                if h_list == []:
                    h_list = h_list_batch
                else:
                    for j, (batch_h, prev_h) in enumerate(zip(h_list, h_list_batch)):
                        h_list[j] = batch_h + prev_h

                logger.info("Computed batch %d for the exact hessian...", i)

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the exact hessian...")
            for i, h in enumerate(h_list):
                if self.damping == "identity":
                    h_list[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_list[i] = h + self.mu * h.diagonal()
                else:
                    raise ValueError(f"Invalid damping strategy {self.damping}.")

        return h_list

    def hvp(self, step_dir) -> Iterable:
        logger.info("Computing the product p^T H p.")

        loss_fn = copy(self.loss_fn_)
        is_mean = loss_fn.reduction == "mean"
        if is_mean:
            loss_fn.reduction = "sum"

        scale = 1 / len(self.x_) if is_mean else 1

        def eval_model_batch(*input_params, x, y):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return loss_fn(out, y)

        if self.batch_size is None or self.batch_size >= len(self.x_):
            eval_model = partial(eval_model_batch, x=self.x_, y=self.y_)
            _, hess_dot_step = torch.autograd.functional.hvp(eval_model, self.params, v=tuple(step_dir))
            hess_dot_step = tuple(hv_i * scale for hv_i in hess_dot_step)
        else:
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the exact hessian vector product split in %d batches of size %d.", len(batch_start), self.batch_size)

            hess_dot_step = None
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate hessian of the batch
                eval_model = partial(eval_model_batch, x=x_batch, y=y_batch)

                _, hess_dot_step_batch = torch.autograd.functional.hvp(eval_model, self.params, v=tuple(step_dir))

                if hess_dot_step is None:
                    hess_dot_step = tuple(batch_hv_i * scale for batch_hv_i in hess_dot_step_batch)
                else:
                    hess_dot_step = tuple(hv_i + batch_hv_i * scale for hv_i, batch_hv_i in zip(hess_dot_step, hess_dot_step_batch))
                logger.info("Computed batch %d for the exact hessian vector product...", i)

        return hess_dot_step

    def quadratic_form(self, d_p_list: Iterable[torch.Tensor]) -> torch.Tensor:
        scaling_matrix_dot_grad = self.hvp(d_p_list)
        quadratic_form = sum(torch.sum(vi * hvi) for vi, hvi in zip(d_p_list, scaling_matrix_dot_grad))

        return quadratic_form
