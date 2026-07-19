Available optimization algorithms
=================================

The library provides concrete optimizers that combine a curvature estimator with a step-selection strategy. They are grouped into three families:

- **Fixed step size** – a scalar learning rate (possibly initialized adaptively) is used directly.
- **Line search** – a one-dimensional search (backtracking, interpolation, or bisection) determines the step length.
- **Trust region** – the step is constrained to a region where the quadratic model is trusted; the radius is updated dynamically.

All optimizers inherit from :class:`torch.optim.Optimizer` and are used with a closure-based objective
(see :class:`~torch_numopt.objective.ObjectiveFunction`).


Fixed-step optimizers
---------------------

These optimizers compute a direction and apply a scalar step length. The learning rate can be set directly via
``lr_init`` or initialized adaptively using ``lr_method`` (e.g., ``"BB1"``, ``"lipschitz"``, ``"quadratic"``).

.. list-table::
   :header-rows: 1

   * - Class
     - Curvature used
     - Algorithm-specific parameters
     - Notes

   * - :py:class:`~torch_numopt.algorithms.gradient_descent.GradientDescent`
     - Identity
     - ``lr_init``, ``lr_method``
     - Standard full-batch gradient descent.

   * - :py:class:`~torch_numopt.algorithms.gradient_descent.GradientDescentLipschitz`
     - Identity
     - ``lr_init``
     - Parameter-free; estimates the Lipschitz constant to set the step size.

   * - :py:class:`~torch_numopt.algorithms.conjugate_gradient.ConjugateGradient`
     - Identity
     - ``lr_init``, ``lr_method``, ``cg_method``
     - Non-linear CG (Fletcher-Reeves, Polak-Ribière, etc.). ``cg_method`` can be ``"FR"``, ``"PR"``, ``"PRP+"``, ``"HS"``, or ``"DY"``.

   * - :py:class:`~torch_numopt.algorithms.newton.Newton`
     - Exact Hessian
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``block_hessian``, ``solver``
     - Full or block-diagonal exact Newton. Damping (identity/Fletcher) is strongly recommended.

   * - :py:class:`~torch_numopt.algorithms.newton.NewtonCG`
     - Exact Hessian (Hvp)
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``solver``
     - Inexact Newton using an iterative solver (e.g., ``"cg-trunc"``). Never forms the full Hessian.

   * - :py:class:`~torch_numopt.algorithms.hutchinson_newton.DiagonalNewton`
     - Diagonal Hessian (Hutchinson)
     - ``lr_init``, ``lr_method``, ``n_samples``, ``skip_iters``
     - Newton's method using the Hutchinson Diagonal approximation. Not recommended but added for completeness

   * - :py:class:`~torch_numopt.algorithms.gauss_newton.GaussNewton`
     - Gauss-Newton (JᵀJ)
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``block_hessian``, ``solver``
     - For least-squares problems. Uses the JᵀJ approximation.

   * - :py:class:`~torch_numopt.algorithms.lbfgs.LBFGS`
     - Inverse Hessian (L-BFGS)
     - ``lr_init``, ``lr_method``, ``memory_size``
     - Limited-memory BFGS with a fixed step size (requires careful tuning).

   * - :py:class:`~torch_numopt.algorithms.adahessian.AdaHessian`
     - Diagonal Hessian (Hutchinson)
     - ``lr_init``, ``lr_method``, ``beta1``, ``beta2``, ``k``, ``eps``, ``n_samples``, ``skip_iters``
     - Adaptive method similar to Adam, but using the diagonal of the Hessian.


Line-search optimizers
----------------------

These optimizers compute a direction and then perform a line search to find an acceptable step length.

.. note::
   All ``*LS`` optimizers accept the standard line-search parameters:
   ``c1``, ``c2``, ``tau``, ``max_iter``, ``tol``, ``line_search_method``, and ``line_search_cond``.
   The table below lists only the parameters that are specific to each algorithm.

.. list-table::
   :header-rows: 1

   * - Class
     - Curvature used
     - Algorithm-specific parameters
     - Notes

   * - :py:class:`~torch_numopt.algorithms.gradient_descent.GradientDescentLS`
     - Identity
     - ``lr_init``, ``lr_method``
     - Gradient descent with robust step-size selection.

   * - :py:class:`~torch_numopt.algorithms.conjugate_gradient.ConjugateGradientLS`
     - Identity
     - ``lr_init``, ``lr_method``, ``cg_method``
     - Conjugate gradient with line search (recommended for CG).

   * - :py:class:`~torch_numopt.algorithms.newton.NewtonLS`
     - Exact Hessian
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``block_hessian``, ``solver``
     - Newton method with line search; much more robust than the fixed-step version.

   * - :py:class:`~torch_numopt.algorithms.newton.NewtonCGLS`
     - Exact Hessian (Hvp)
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``solver``
     - Inexact Newton with line search (``solver`` must be iterative).

   * - :py:class:`~torch_numopt.algorithms.gauss_newton.GaussNewtonLS`
     - Gauss-Newton (JᵀJ)
     - ``lr_init``, ``lr_method``, ``damping``, ``mu``, ``block_hessian``, ``solver``
     - Gauss-Newton with line search; preferred over the fixed-step variant.

   * - :py:class:`~torch_numopt.algorithms.lbfgs.LBFGSLS`
     - Inverse Hessian (L-BFGS)
     - ``lr_init``, ``lr_method``, ``memory_size``
     - L-BFGS with line search (typically used with Wolfe conditions). This is the standard way to use L-BFGS.

   * - :py:class:`~torch_numopt.algorithms.adahessian.AdaHessianLS`
     - Diagonal Hessian (Hutchinson)
     - ``lr_init``, ``lr_method``, ``beta1``, ``beta2``, ``k``, ``eps``, ``n_samples``, ``skip_iters``
     - AdaHessian with line search.

   * - :py:class:`~torch_numopt.algorithms.adahessian.DiagonalNewtonLS`
     - Diagonal Hessian (Hutchinson)
     - ``lr_init``, ``lr_method``, ``n_samples``, ``skip_iters``
     - Diagonal Newton with line search.


Trust-region optimizers
-----------------------

These optimizers solve a subproblem that restricts the step to a region where the quadratic model is trusted.

.. note::
   All ``*TR`` optimizers accept the common trust-region parameters:
   ``trust_region_method``, and ``accept_tol``, ``contract_tol``, ``expand_tol``, ``growth_factor``, ``shrink_factor``, ``radius_max``.
   The table below lists only the parameters that are specific to each algorithm.

.. list-table::
   :header-rows: 1

   * - Class
     - Curvature used
     - Algorithm-specific parameters
     - Notes

   * - :py:class:`~torch_numopt.algorithms.gradient_descent.GradientDescentTR`
     - Identity
     - (none beyond the common ones)
     - Gradient descent with trust region (Cauchy point).

   * - :py:class:`~torch_numopt.algorithms.newton.NewtonTR`
     - Exact Hessian
     - ``damping``, ``mu``, ``block_hessian``, ``solver``
     - Newton method with trust region. Supports exact and Steihaug-Toint solvers.

   * - :py:class:`~torch_numopt.algorithms.newton.NewtonCGTR`
     - Exact Hessian (Hvp)
     - ``damping``, ``mu``
     - Inexact Newton with trust region (Steihaug-Toint). Memory-efficient; only uses Hvp.

   * - :py:class:`~torch_numopt.algorithms.gauss_newton.GaussNewtonTR`
     - Gauss-Newton (JᵀJ)
     - ``damping``, ``mu``, ``block_hessian``, ``solver``
     - Gauss-Newton with trust region; robust and efficient for least-squares.

   * - :py:class:`~torch_numopt.algorithms.levenberg_marquardt.LevenbergMarquardt`
     - Damped Gauss-Newton
     - ``mu``, ``mu_max``, ``solver``
     - Interpolates between Gauss-Newton and gradient descent via adaptive damping. Uses block Gauss-Newton with Fletcher damping internally.

   * - :py:class:`~torch_numopt.algorithms.levenberg_marquardt.InexactLevenbergMarquardt`
     - Damped Gauss-Newton
     - ``mu``, ``mu_max``, ``solver``
     - Interpolates between Gauss-Newton and gradient descent via adaptive damping. Uses inexact solvers to solve the step size subproblem.