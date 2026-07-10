.. _custom-optimizers:

Custom optimizers: building from components
===========================================

While the library provides many ready-to-use optimizers (see :doc:`/api_reference/algorithms`), you can also **assemble your own** by combining:

- a :class:`~torch_numopt.curvature_estimator.CurvatureEstimator` (defines the Hessian approximation),
- and a **step-length strategy**. The library supports three distinct strategies:

  1. **Fixed step size** – a scalar learning rate (possibly initialized adaptively using ``lr_method``).  
     Use :class:`~torch_numopt.numerical_optimizer.NumericalOptimizer` directly.

  2. **Line search** – a one-dimensional search (backtracking, interpolation, or bisection) that determines the step length.  
     Use :class:`~torch_numopt.numerical_optimizer.LineSearchOptimizer` with a :class:`~torch_numopt.line_search.LineSearchSolver`.

  3. **Trust region** – the step is constrained to a region where the quadratic model is trusted; the radius is updated dynamically.  
     Use :class:`~torch_numopt.numerical_optimizer.TrustRegionOptimizer` with a :class:`~torch_numopt.trust_region.TrustRegionSolver`.

This gives you full control over the algorithm, while still benefiting from the library's robust infrastructure (history management, numerical stability, linear solvers, etc.).

For convenience, factory functions are provided for line-search and trust-region solvers:
:func:`~torch_numopt.line_search.create_line_search_solver` and
:func:`~torch_numopt.trust_region.create_trust_region_solver`.


Available curvature estimators
------------------------------

All curvature estimators inherit from :class:`~torch_numopt.curvature_estimator.CurvatureEstimator`
and must implement the Hessian-vector product and quadratic form.

.. list-table::
   :header-rows: 1

   * - Class
     - Representation
     - Parameters
     - Description

   * - :class:`~torch_numopt.curvature.NaiveIdentityCalculator`
     - scalar (0-D)
     - (none)
     - Identity curvature → first-order methods.

   * - :class:`~torch_numopt.curvature.HutchinsonDiagonalApproximation`
     - vector (1-D)
     - ``n_samples`` (int, default=1)
     - Diagonal Hessian estimated via Hutchinson's method (unbiased).

   * - :class:`~torch_numopt.curvature.ExactBlockHessianCalculator`
     - tuple of matrices (block-diagonal)
     - ``damping``, ``mu``
     - Exact Hessian, but only diagonal blocks (one per parameter group).

   * - :class:`~torch_numopt.curvature.ExactHessianCalculator`
     - single dense matrix (2-D)
     - ``damping``, ``mu``
     - Exact full Hessian (can be very large).

   * - :class:`~torch_numopt.curvature.GaussNewtonBlockApproximation`
     - tuple of matrices (block-diagonal)
     - ``vectorize``, ``damping``, ``mu``
     - JᵀJ approximation, block-diagonal.

   * - :class:`~torch_numopt.curvature.GaussNewtonApproximation`
     - single dense matrix (2-D)
     - ``vectorize``, ``damping``, ``mu``
     - Full JᵀJ approximation.

.. note::
   - Damping (``"identity"`` or ``"fletcher"``) adds a multiple of the identity (or diagonal) to improve conditioning.
   - The ``vectorize`` parameter for Gauss-Newton enables faster Jacobian computation via :func:`torch.func.jacrev`.

Step-size initialization methods
--------------------------------

For fixed-step and line-search optimizers, the initial learning rate is determined by the
``lr_init`` parameter, but it can be adjusted using a heuristic via the ``lr_method`` argument.
These heuristics use information from the previous iteration (gradients, step directions, and
loss changes) to propose a better starting point for the step size.

The following methods are available:

.. list-table::
   :header-rows: 1

   * - ``lr_method``
     - Description

   * - ``None``
     - Use ``lr_init`` directly (no adjustment).

   * - ``"keep"``
     - Reuse the learning rate from the previous iteration.

   * - ``"scaled"``
     - Scale the initial rate based on the dot product of previous gradient and step direction
       relative to the current one.

   * - ``"quadratic"``
     - Use the quadratic form of the curvature to estimate a step that minimizes the model:
       :math:`\alpha = - (g^T p) / (p^T H p)`.

   * - ``"interpolate"``
     - Estimate the rate from the change in loss between iterations (useful after a line search).

   * - ``"lipschitz"``
     - Estimate the Lipschitz constant from the change in gradients and parameters:
       :math:`\alpha = \|s\| / \|y\|`.

   * - ``"BB1"``
     - Barzilai-Borwein formula 1: :math:`\alpha = (s^T s) / (s^T y)`.

   * - ``"BB2"``
     - Barzilai-Borwein formula 2: :math:`\alpha = (s^T y) / (y^T y)`.

where :math:`s` is the previous step (parameter change) and :math:`y` is the previous gradient
change. These heuristics are particularly useful for gradient descent and Newton-type methods
with fixed step sizes, as they can accelerate convergence without manual tuning.

Available line-search solvers
------------------------------

Line-search solvers find a step length that satisfies a chosen condition. They are used with :class:`~torch_numopt.numerical_optimizer.LineSearchOptimizer`.

.. list-table::
   :header-rows: 1

   * - Class
     - Factory identifier (``method``)
     - Parameters
     - Description

   * - :class:`~torch_numopt.line_search.BacktrackingLineSearch`
     - ``"backtrack"``
     - ``condition``, ``c1``, ``c2``, ``tau``, ``max_iter``, ``tol``
     - Repeatedly reduces the step by factor ``tau`` until the condition holds.

   * - :class:`~torch_numopt.line_search.InterpolationLineSearch`
     - ``"interpolate"``
     - ``condition``, ``c1``, ``c2``, ``tau``, ``max_iter``, ``tol``
     - Fits a quadratic/cubic model to estimate a better step.

   * - :class:`~torch_numopt.line_search.BisectionLineSearch`
     - ``"bisect"``
     - ``condition``, ``c1``, ``c2``, ``tau``, ``max_iter``, ``tol``
     - Maintains a bracketing interval and narrows it using derivative signs.

**Conditions** (the ``condition`` parameter) are:
- ``"greedy"`` – only requires decrease.
- ``"armijo"`` – sufficient decrease (default).
- ``"wolfe"`` – Armijo + curvature condition.
- ``"strong-wolfe"`` – Armijo + strong curvature condition.
- ``"goldstein"`` – Goldstein conditions.

The factory function :func:`~torch_numopt.line_search.create_line_search_solver` simplifies instantiation:

.. code-block:: python

   ls_solver = create_line_search_solver(
       method="interpolate",      # one of the identifiers above
       condition="wolfe",
       c1=1e-4,
       c2=0.9,
       tau=0.5,
       max_iter=30,
       tol=1e-10
   )


Available trust-region solvers
-------------------------------

Trust-region solvers solve the subproblem :math:`\min_{||p|| \le \Delta} m(p)`.
They are used with :class:`~torch_numopt.numerical_optimizer.TrustRegionOptimizer`.

.. list-table::
   :header-rows: 1

   * - Class
     - Factory identifier (``method``)
     - Parameters
     - Description

   * - :class:`~torch_numopt.trust_region.CauchyPointTRSolver`
     - ``"cauchy"``
     - ``curvature_estimator``, ``solver`` (ignored)
     - Minimises along the steepest descent direction; cheap and guaranteed decrease.

   * - :class:`~torch_numopt.trust_region.DoglegTRSolver`
     - ``"dogleg"``
     - ``curvature_estimator``, ``solver``
     - Piecewise linear path combining steepest descent and Newton steps.

   * - :class:`~torch_numopt.trust_region.ExactTRSolver`
     - ``"exact"``
     - ``curvature_estimator``, ``iters``, ``tol``
     - Solves the subproblem exactly via Cholesky; expensive but accurate.

   * - :class:`~torch_numopt.trust_region.SteihaugTointTRSolver`
     - ``"steihaug-toint"``
     - ``curvature_estimator``, ``max_iter``, ``atol``, ``tol``, ``min_iter``
     - Uses conjugate gradients; memory-efficient and recommended for large problems.

The factory function :func:`~torch_numopt.trust_region.create_trust_region_solver`
provides a convenient way to instantiate:

.. code-block:: python

   tr_solver = create_trust_region_solver(
       method="steihaug-toint",   # one of the identifiers above
       curvature_estimator=my_curvature,
       solver="cg-trunc",
       max_iter=50
   )


Available linear solvers
------------------------

When the curvature estimator returns a matrix (full or block), the system :math:`H p = b` must be solved.
The solver is selected via the ``solver`` parameter (passed to the optimizer or to the trust-region solver).

.. list-table::
   :header-rows: 1

   * - Solver name (``solver``)
     - Type
     - Description

   * - ``"solve"``
     - LU (Direct)
     - Standard Gaussian elimination.

   * - ``"cholesky"``
     - Cholesky (Direct)
     - Requires positive-definite matrix; faster than LU.

   * - ``"pinv"``
     - Pseudo-inverse (Direct)
     - Moore-Penrose inverse; handles singular matrices.

   * - ``"pinv-trunc"``
     - Truncated SVD (Direct)
     - Pseudo-inverse with small singular values truncated for stability.

   * - ``"lsqrs"``
     - Least-squares (Direct)
     - Solves in the least-squares sense.

   * - ``"safe-lsqrs"``
     - Robust least-squares (Direct)
     - Uses a more robust driver.

   * - ``"cg"``
     - Conjugate gradient (Iterative)
     - For large systems; requires Hvp.

   * - ``"cg-trunc"``
     - Truncated CG (Iterative)
     - Stops on negative curvature; used in Newton-CG.

   * - ``"cr"``
     - Conjugate residual (Iterative)
     - Similar to CG but can be more robust.


Building a custom optimizer: step-by-step
-----------------------------------------

1. **Choose a curvature estimator** – e.g., ``ExactBlockHessianCalculator(damping="identity", mu=1e-4)``.

2. **Choose a step-length strategy**:

   - **Fixed step**: set a learning rate (and optionally an adaptive initialization method). No additional solver object is required.
   - **Line search**: instantiate or create a :class:`~torch_numopt.line_search.LineSearchSolver`.
   - **Trust region**: instantiate or create a :class:`~torch_numopt.trust_region.TrustRegionSolver`.

3. **Instantiate the appropriate base optimizer**:

   - For **fixed step**:  
     ``NumericalOptimizer(params, curvature_estimator, lr_init, lr_method, solver, ...)``

   - For **line search**:  
     ``LineSearchOptimizer(params, curvature_estimator, line_search, lr_init, lr_method, solver)``

   - For **trust region**:  
     ``TrustRegionOptimizer(params, trust_region, radius_init, accept_tol)``

Example 1: fixed-step diagonal Newton with adaptive LR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch_numopt import NumericalOptimizer, HutchinsonDiagonalApproximation

   curvature = HutchinsonDiagonalApproximation(n_samples=5)

   optimizer = NumericalOptimizer(
       params=model.parameters(),
       curvature_estimator=curvature,
       lr_init=1.0,
       lr_method="lipschitz",   # adaptive initialisation, no line search
       solver="solve"
   )

Example 2: custom Gauss-Newton with dogleg trust region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch_numopt import (
       TrustRegionOptimizer,
       GaussNewtonBlockApproximation,
       DoglegTRSolver
   )

   curvature = GaussNewtonBlockApproximation(damping="identity", mu=1e-3)
   tr_solver = DoglegTRSolver(curvature_estimator=curvature, solver="cholesky")

   optimizer = TrustRegionOptimizer(
       params=model.parameters(),
       trust_region=tr_solver,
       radius_init=1.0,
       accept_tol=0.1
   )

Example 3: custom diagonal Newton with interpolation line search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from torch_numopt import (
       LineSearchOptimizer,
       HutchinsonDiagonalApproximation,
       create_line_search_solver
   )

   curvature = HutchinsonDiagonalApproximation(n_samples=5)

   ls_solver = InterpolationLineSearch(
       condition="armijo",
       c1=1e-4,
   )

   optimizer = LineSearchOptimizer(
       params=model.parameters(),
       curvature_estimator=curvature,
       line_search=ls_solver,
       lr_init=1.0,
       lr_method="lipschitz",
       solver="solve"
   )