""" """

from __future__ import annotations
from typing import Iterable
from abc import ABC, abstractmethod
import logging
import torch
from torch import nn
from functools import reduce, partial
from .utils import param_reshape_like
from torch.func import functional_call
from copy import copy

logger = logging.getLogger(__name__)

class ScalingMatrixCalculator(ABC):
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
    
    def store_data(self, x, y, loss_fn):
        """
        Stores the necessary data for later use
        """

        self.x_ = x
        self.y_ = y
        self.loss_fn_ = loss_fn

    @abstractmethod
    def scaling_matrix(self) -> Iterable | None:
        """
        Obtains the second derivative approximation.
        """
    
    @abstractmethod
    def hvp(self, step_dir) -> Iterable | None:
        """
        Compute p B_k p^T
        with B being the scaling matrix and p the step direction
        """


class NaiveIdentityCalculator(ScalingMatrixCalculator):
    """
    Naive second derivative approximator. Always assumes an identity as the hessian.
    """
    def scaling_matrix(self) -> None:
        return None
    
    def hvp(self, step_dir) -> Iterable | None:
        return step_dir

class ExactBlockHessianCalculator(ScalingMatrixCalculator):
    """
    Approximates the hessian in blocks, only taking the inner-layer second derivatives.
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int | None = None,
        damping: str | None = None,
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

        def eval_model_batch(x, y, *input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the exact hessian matrix.")

            # Calculate hessian with every sample in the dataset
            eval_model = partial(eval_model, x=self.x_, y=self.y_)

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
                    h_list_batch[j] = self._reshape_hessian(h_list_batch[i][i]) * scale

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

    def hvp(self, step_dir) -> Iterable | None:
        logger.info("Computing the product p^T H p.")

        loss_fn = copy(self.loss_fn_)
        is_mean = loss_fn.reduction == "mean"
        if is_mean:
            loss_fn.reduction = "sum"

        scale = 1 / len(self.x_) if is_mean else 1

        def eval_model_batch(x, y, *input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return loss_fn(out, y)

        if self.batch_size is None or self.batch_size >= len(self.x_):
            eval_model = partial(eval_model_batch, x=self.x_, y=self.y_)
            _, hess_dot_step = torch.autograd.functional.hvp(eval_model, self.params, v=tuple(step_dir))
        else:
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the exact hessian vector product split in %d batches of size %d.", len(batch_start), self.batch_size)

            hess_dot_step = 0
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate hessian of the batch
                eval_model = partial(eval_model_batch, x=x_batch, y=y_batch)

                _, hess_dot_step_batch = torch.autograd.functional.hvp(eval_model, self.params, v=tuple(step_dir))

                hess_dot_step += hess_dot_step_batch
                logger.info("Computed batch %d for the exact hessian...", i)

        return hess_dot_step * scale


class GaussNewtonBlockApproximation(ScalingMatrixCalculator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int | None = None,
        vectorize: bool = True,
        damping: str | None = None,
        mu: float = 1e-4,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.vectorize = vectorize
        self.damping = damping
        self.mu = mu

    def scaling_matrix(self) -> Iterable:
        r"""
        Calculation of the an approximate hessian of the Neural network given a dataset as in the Gauss-Newton algorithm.
        The approximate Hessian is calculated as the square of the Jacobian of the residual of every data point with respect to the parameters.

        Let the loss function be, for example the MSE:

        :math:`\mathcal{L}(x,y;\theta) = \sum^{N}_{i=1} (f(x_i; \theta) - y_i)^2 = \sum^{N}_{i=1} r_i`

        Then the Jacobian of the residuals will be the matrix:

        :math:`(J_{\theta}[\mathcal{L}])_{i,j} = \dfrac{\partial r_i}{\partial \theta_j}`

        Then, we will approximate the hessian as the product of the Jacobian with it's transpose, noting that the result
        will be a square matrix with size :math:`p\\times p` with :math:`p` being the number of parameters of the model:

        :math:`H_{\theta}[\mathcal{L}] \approx J_{\theta}[\mathcal{L}]^{\intercal} \cdot J_{\theta}[\mathcal{L}]`

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
        model_params = tuple(self.model.parameters())

        scale = 2 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        residual_fn = copy(self.loss_fn_)
        residual_fn.reduction = "none"

        def get_residuals_batch(x, y, *input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return residual_fn(out, y)

        # Calculate approximate Hessian matrix
        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Gauss-Newton approximate Hessian matrix.")
            # get_residuals = lambda *p: get_residuals_batch(x, y, *p)
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=self.vectorize)
            h_list = [None] * len(j_list)
            for j_idx, j in enumerate(j_list):
                j = j.view(j.shape[0], -1)
                h_list[j_idx] = self._reshape_hessian(j.T @ j) * scale
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", len(batch_start), self.batch_size)

            h_list = []
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=self.vectorize)
                h_list_batch = [None] * len(j_list)
                for j_idx, j in enumerate(j_list):
                    j = j.view(j.shape[0], -1)
                    h_list_batch[j_idx] = self._reshape_hessian(j.T @ j) * scale

                # Aggregate result
                if h_list == []:
                    h_list = h_list_batch
                else:
                    h_list = [batch_h + prev_h for batch_h, prev_h in zip(h_list, h_list_batch)]

                logger.info("Computed batch %d for the approximate hessian...", i)

        # Damp matrix
        if self.damping is not None:
            logger.info("Applying damping to the approximate hessian...")
            for i, h in enumerate(h_list):
                if self.damping == "identity":
                    h_list[i] = h + self.mu * torch.eye(h.shape[0], device=h.device)
                elif self.damping == "fletcher":
                    h_list[i] = h + self.mu * h.diagonal()
                else:
                    raise ValueError("Invalid damping strategy.")

        return h_list

    def hvp(self, step_dir):
        model_params = tuple(self.model.parameters())

        scale = 2 / len(self.x_) if self.loss_fn_.reduction == "mean" else 1

        residual_fn = copy(self.loss_fn_)
        residual_fn.reduction = "none"

        def get_residuals_batch(x, y, *input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), x)
            return residual_fn(out, y)

        if self.batch_size is None or self.batch_size >= len(self.x_):
            logger.info("Computing the Gauss-Newton approximate Hessian matrix.")
            get_residuals = partial(get_residuals_batch, x=self.x_, y=self.y_)
            jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, tangents=step_dir, v)
        else:
            # Calculate hessian for each batch and add the results
            batch_start = torch.arange(0, len(self.x_), self.batch_size)

            logger.info("Computing the Gauss-Newton approximate Hessian matrix split in %d batches of size %d.", len(batch_start), self.batch_size)

            h_list = []
            for i, start in enumerate(batch_start):
                # Prepare batch
                x_batch = self.x_[start : start + self.batch_size]
                y_batch = self.y_[start : start + self.batch_size]

                # Calculate approximate hessian of the batch
                get_residuals = partial(get_residuals_batch, x=x_batch, y=y_batch)
                jac_dot_step = torch.autograd.functional.jvp(get_residuals, model_params, tangents=step_dir, v)

                hess_dot_step += hess_dot_step_batch
                logger.info("Computed batch %d for the Gauss-newton approximate hessian...", i)

        return hess_dot_step * scale

class HutchinsonDiagonalApproximation(ScalingMatrixCalculator):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = None,
        n_samples: int = 1,
    ):
        super().__init__(model=model, batch_size=batch_size)
        self.n_samples = n_samples

    def scaling_matrix(self) -> Iterable:
        model_params = tuple(self.model.parameters())
        params_flat = torch.hstack([i.ravel() for i in model_params])

        def eval_model(*input_params):
            out = functional_call(self.model, dict(zip(self.param_keys, input_params)), self.x_)
            return self.loss_fn_(out, self.y_)

        h_diag_flat = torch.zeros_like(params_flat)
        logger.info("Computing diagonal Hutchinson approximation of the hessian with %d samples.", self.n_samples)
        for i in range(self.n_samples):
            # Rademacher sample
            z_flat = 2 * torch.bernoulli(torch.full_like(params_flat, 0.5, device=params_flat.device)) - 1
            z = tuple(param_reshape_like(z_flat, model_params))

            # Pytorch documentation recommends doing (vH)^T instead of Hv directly
            _, Hz = torch.autograd.functional.vhp(eval_model, model_params, v=z, create_graph=False)
            Hz_flat = torch.hstack([i.ravel() for i in Hz])

            h_diag_flat += z_flat * Hz_flat

            logger.info("Calculated approximation for random sample number %d...", i)
        h_diag_flat /= self.n_samples

        h_diag = param_reshape_like(h_diag_flat, model_params)
        return h_diag

    def hvp(self, step_dir):
        diag_hessian = self.scaling_matrix
        return tuple(p * h for p, h in zip(step_dir, diag_hessian))
