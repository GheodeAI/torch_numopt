from __future__ import annotations
import torch
import torch.nn as nn

from ..line_search import create_line_search_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import GaussNewtonBlockApproximation
from ..utils import Params, param_add, param_scaled_add, param_dot, param_scalar_prod


class LevenbergMarquardt(NumericalOptimizer):
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
        mu_dec: float = 0.01,
        mu_max: float = 1e10,
        fletcher: bool = False,
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        self.fletcher = fletcher
        damping = "fletcher" if fletcher else "identity"

        super().__init__(
            params,
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            solver=solver,
        )

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max
        self.prev_loss = None

    def step(self, objective):
        super().step(objective)
        self.update(objective)

    def update(self, objective):
        if self.prev_loss is None:
            super().update(objective)
            return

        pred_step = param_scalar_prod(-self.curr_lr, self.curr_step_dir)
        pred_reduction = - (0.5 * self.curr_lr - 1) * param_dot(pred_step, self.prev_grad)

        rho = (self.prev_loss - self.curr_loss) / pred_reduction

        if rho > 0:
            self.mu *= self.mu_dec
            super().update(objective)
        else:
            with torch.no_grad():
                for p, prev_p in zip(self.curr_params, self.prev_params):
                    p.copy_(prev_p)
            self.mu /= self.mu_dec

        if self.mu >= self.mu_max:
            self.mu = self.mu_max

        self.curvature_estimator.mu = self.mu



class LevenbergMarquardtLS(LineSearchOptimizer):
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
        fletcher: bool = False,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        self.fletcher = fletcher
        damping = "fletcher" if fletcher else "identity"

        super().__init__(
            params,
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
            solver=solver,
        )

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max
        self.prev_loss = None

    def step(self, objective):
        super().step(objective)
        self.update(objective.loss(self.params))

    def update(self, loss: torch.Tensor):
        loss_val = loss.detach().item()

        if self.prev_loss is None:
            self.prev_loss = loss_val
            self.prev_params_ = [p.detach().clone() for p in self.params]
        elif loss_val <= self.prev_loss:
            self.prev_loss = loss_val
            self.prev_params_ = [p.detach().clone() for p in self.params]
            self.mu *= self.mu_dec
        else:
            self.params = self.prev_params_
            self.mu /= self.mu_dec

        if self.mu >= self.mu_max:
            self.mu = self.mu_max

        self.curvature_estimator.mu = self.mu


class LevenbergMarquardtTR(TrustRegionOptimizer):
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
        fletcher: bool = False,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
        solver: str = "solve",
        batch_size: int | None = None,
    ):
        self.fletcher = fletcher
        damping = "fletcher" if fletcher else "identity"

        super().__init__(
            params,
            curvature_estimator=GaussNewtonBlockApproximation(damping=damping, mu=mu),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau),
            solver=solver,
        )

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max
        self.prev_loss = None

    # def update_model_radius(self):
    #     # loss_val = loss.detach().item()
    #     if self.prev_loss is None:
    #         self.prev_loss = loss_val
    #         self._prev_params = [p.detach().clone() for p in self._params]
    #     elif loss_val <= self.prev_loss:
    #         self.prev_loss = loss_val
    #         self._prev_params = [p.detach().clone() for p in self._params]
    #         self.mu *= self.mu_dec
    #     else:
    #         self._params = self._prev_params
    #         self.mu /= self.mu_dec

    #     if self.mu >= self.mu_max:
    #         self.mu = self.mu_max

    #     self.curvature_estimator.mu = self.mu
