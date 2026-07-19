"""
Hutchinson diagonal Hessian approximation.

This estimator uses random Rademacher vectors and the Hessian-vector product to
estimate the diagonal of the Hessian. It is unbiased and works well for large
models where full Hessian computation is infeasible.
"""

from __future__ import annotations
import logging
import torch
from ..utils import param_reshape_like, param_numel, param_scalar_prod, param_mult, param_add, param_zero_like, param_dot, Params
from ..curvature_estimator import CurvatureEstimator
from ..objective import ObjectiveFunction

logger = logging.getLogger(__name__)


class HutchinsonDiagonalApproximation(CurvatureEstimator):
    """
    Diagonal Hessian estimator via Hutchinson's method.

    The diagonal is estimated as the average of z ⊙ (H z) over random Rademacher
    vectors z.

    Parameters
    ----------
    n_samples : int, default=1
        Number of random samples to average.
    """

    def __init__(self, n_samples: int = 1, skip_iters: int = 0):
        super().__init__(ndim=1, uses_blocks=True)
        self.n_samples = n_samples
        self.skip_iters = skip_iters
        self.iter_counter = 0
        self.stale_diagonal = True
        self.stored_diagonal = None

    def scaling_matrix(self, objective: ObjectiveFunction, params: Params) -> Params:
        if not self.stale_diagonal and self.stored_diagonal is not None:
            logger.debug("Used stored diagonal.")
            return self.stored_diagonal

        param_size = param_numel(params)
        device = params[0].device
        h_diag = param_zero_like(params)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Computing diagonal Hutchinson approximation of the hessian with %d samples.", self.n_samples)

        for i in range(self.n_samples):
            # Rademacher sample
            z_flat = 2 * torch.randint(0, 2, size=(param_size,), device=device, dtype=torch.uint8).float() - 1
            z = param_reshape_like(z_flat, params)

            # Pytorch documentation recommends doing (vH)^T instead of Hv directly
            _, Hz = torch.autograd.functional.vhp(objective.loss, params, v=z, create_graph=False)

            h_diag = param_add(h_diag, param_mult(z, Hz))

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Calculated approximation for random sample number %d...", i)

        h_diag = param_scalar_prod(1 / self.n_samples, h_diag)
        self.stored_diagonal = h_diag
        self.stale_diagonal = False

        return h_diag

    def hvp(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> Params:
        if self.stored_diagonal is None or self.stale_diagonal:
            diagonal = self.scaling_matrix(objective, params)
        else:
            diagonal = self.stored_diagonal

        return param_dot(diagonal, step_dir)

    def quadratic_form(self, objective: ObjectiveFunction, params: Params, step_dir: Params) -> torch.Tensor:
        return param_dot(step_dir, self.hvp(objective, params, step_dir))

    def update(self):
        self.iter_counter += 1
        if self.iter_counter > self.skip_iters:
            self.iter_counter = 0
            self.stale_diagonal = True
        logger.debug("Updated iteration counter, stale diagonal? %d.", self.stale_diagonal)
