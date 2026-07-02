"""
Gradient descent optimizers (first-order).

These optimizers use only the gradient (identity curvature). They are the
simplest methods and serve as a baseline.
"""

from __future__ import annotations
from typing import Iterable
import torch
from ..line_search import create_line_search_solver
from ..trust_region import create_trust_region_solver
from ..numerical_optimizer import NumericalOptimizer, LineSearchOptimizer, TrustRegionOptimizer
from ..curvature import NaiveIdentityCalculator
from ..utils import Params


class GradientDescent(NumericalOptimizer):
    """
    Vanilla gradient descent with a fixed or adaptively initialized learning rate.

    Parameters
    ----------
    params : Params
        Parameter tensors.
    lr_init : float, default=1e-3
        Initial learning rate.
    lr_method : str, optional
        Learning-rate initialization method (see :class:`NumericalOptimizer`).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr_init: float = 1e-3,
        lr_method: str = None,
    ):

        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
        )


class GradientDescentLS(LineSearchOptimizer):
    """
    Gradient descent with a line-search for step length.
    """

    def __init__(
        self,
        params: Params,
        lr_init: float = 1,
        lr_method: str = None,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        max_iter: int = 20,
        tol: float = 1e-8,
        line_search_method: str = "backtrack",
        line_search_cond: str = "armijo",
    ):

        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method=lr_method,
            line_search=create_line_search_solver(
                method=line_search_method, condition=line_search_cond, c1=c1, c2=c2, tau=tau, max_iter=max_iter, tol=tol
            ),
        )


class GradientDescentTR(TrustRegionOptimizer):
    """
    Gradient descent with a trust-region (Cauchy point) step.
    """

    def __init__(
        self,
        params: Params,
        radius_init: float = 1.0,
        trust_region_method: str = "cauchy",
    ):
        super().__init__(
            params,
            trust_region=create_trust_region_solver(method=trust_region_method, curvature_estimator=NaiveIdentityCalculator()),
            radius_init=radius_init,
        )


class GradientDescentLipschitz(NumericalOptimizer):
    """
    Gradient descent with a learning rate estimated from the Lipschitz constant.

    This is a very simple optimizer that works surprisingly well in practice.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr_init: float = 1e-3,
    ):
        super().__init__(
            params=params,
            curvature_estimator=NaiveIdentityCalculator(),
            lr_init=lr_init,
            lr_method="lipschitz",
        )
