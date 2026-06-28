from __future__ import annotations
import torch

from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import GaussNewtonBlockApproximation
from ..utils import Params, param_dot, param_scalar_prod, param_norm, param_copy


class LevenbergMarquardtMixin:
    def __init__(self, *args, mu_dec: float = 0.01, mu_max: float = 1e10, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_dec = mu_dec
        self.mu_max = mu_max

    def update(self):
        if self.prev_loss is None:
            super().update()
            return

        eps = 1e-12

        lr = self.curr_lr
        grad_params = self.curr_grad
        step_dir = self.curr_step_dir
        mu = self.curvature_estimator.mu

        # Correct formula for descent step direction p (g·p < 0)
        pred_reduction = -0.5 * (mu * param_dot(step_dir, step_dir) - param_dot(step_dir, grad_params))

        rho = (self.prev_loss - self.curr_loss) / (pred_reduction + eps)
        print(f"  prev_loss: {self.prev_loss:.6e}, curr_loss: {self.curr_loss:.6e}")
        print(f"  curr_step_dir: norm = {param_norm(self.curr_step_dir):.6e}")
        print(f"  curr_grad: norm = {param_norm(self.curr_grad):.6e}")
        print(f"  mu: {mu:.6e}")
        print(f"  lr: {lr:.6e}")
        print(f"  g_dot_p: {param_dot(self.curr_step_dir, self.curr_grad):.6e}")
        print(f"  p_norm_sq: {param_dot(self.curr_step_dir, self.curr_step_dir):.6e}")
        print(f"  pred_reduction: {pred_reduction:.6e}")
        print(f"  rho: {rho:.6e}")

        if rho > 0:
            super().update()
        else:
            with torch.no_grad():
                for p, prev_p in zip(self.params, self.prev_params):
                    p.copy_(prev_p)

                self.curr_params = param_copy(self.params)
                self.curr_loss = self.prev_loss
                self.curr_grad = self.prev_grad
                self.curr_step_dir = None

        if rho > 0.75:
            self.curvature_estimator.mu *= self.mu_dec
        elif rho < 0.25:
            self.curvature_estimator.mu /= self.mu_dec

        if self.curvature_estimator.mu >= self.mu_max:
            self.curvature_estimator.mu = self.mu_max


class LevenbergMarquardt(LevenbergMarquardtMixin, NumericalOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    and the matlab implementation of 'learnlm' https://es.mathworks.com/help/deeplearning/ref/trainlm.html#d126e69092

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian approximation.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
    use_diagonal: bool
        Whether to use the diagonal of the Hessian approximation instead of an identity matrix to adjust the Hessian matrix.
    c1: float
        Coefficient of the sufficient increase condition in backtracking line search.
    c2: float
        Coefficient used in the second condition for wolfe conditions.
    tau: float
        Factor used to reduce the step size in each step of the backtracking line search.
    line_search_method: str
        Method used for line search, options are "backtrack" and "constant".
    line_search_cond: str
        Condition to be used in backtracking line search, options are "armijo", "wolfe", "strong-wolfe" and "goldstein".
    solver: str
        Method to use to invert the hessian.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1.0,
        lr_method: str | None = None,
        mu: float = 1e-4,
        mu_dec: float = 0.1,
        mu_max: float = 1e10,
        damping: bool = "fletcher",
        solver: str = "solve",
    ):
        super().__init__(
            params,
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
            mu_dec=mu_dec,
            mu_max=mu_max,
            fix_ascent=True,
        )


class LevenbergMarquardtLS(LevenbergMarquardtMixin, LineSearchOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    and the matlab implementation of 'learnlm' https://es.mathworks.com/help/deeplearning/ref/trainlm.html#d126e69092

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr_init: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    lr_method: str
        Method to use to initialize the learning rate before applying line search.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian approximation.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
    use_diagonal: bool
        Whether to use the diagonal of the Hessian approximation instead of an identity matrix to adjust the Hessian matrix.
    c1: float
        Coefficient of the sufficient increase condition in backtracking line search.
    c2: float
        Coefficient used in the second condition for wolfe conditions.
    tau: float
        Factor used to reduce the step size in each step of the backtracking line search.
    line_search_method: str
        Method used for line search, options are "backtrack" and "constant".
    line_search_cond: str
        Condition to be used in backtracking line search, options are "armijo", "wolfe", "strong-wolfe" and "goldstein".
    solver: str
        Method to use to invert the hessian.
    batch_size: int
        Size of the amount of data to use at a time to calculate the hessian matrix.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str | None = None,
        mu: float = 0.001,
        mu_dec: float = 0.1,
        mu_max: float = 1e10,
        damping: str = "fletcher",
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
    ):
        super().__init__(
            params,
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            solver=solver,
            mu_dec=mu_dec,
            mu_max=mu_max,
        )
