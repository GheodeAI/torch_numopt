.. torch_numopt documentation master file, created by
   sphinx-quickstart on Wed Sep  4 15:17:13 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

torch_numopt documentation
==========================

**torch_numopt** is a PyTorch library that provides a comprehensive collection of
**classical numerical optimization algorithms** for machine learning and
general-purpose differentiable optimization. While PyTorch's built-in optimizers
focus on first-order methods (SGD, Adam, etc.), this package implements
second-order and quasi-Newton methods that can significantly accelerate
convergence on problems where curvature information is available or can be
approximated.

Key features
------------

- A wide range of optimizers – Newton (exact Hessian), Gauss-Newton,
  Levenberg-Marquardt, nonlinear Conjugate Gradient, L-BFGS, AdaHessian, and
  gradient descent with line search or trust region.
- Flexible curvature estimators – exact Hessian (full or block-diagonal),
  Gauss-Newton approximation (full or block), diagonal Hessian via Hutchinson's
  method, and identity (for first-order methods).
- Line search and trust-region frameworks – backtracking, interpolation,
  bisection, Cauchy point, dogleg, exact (with Lagrange multiplier), and
  Steihaug-Toint CG.
- Linear solvers – direct (Cholesky, LU, pseudo-inverse, least-squares) and
  iterative (CG, truncated CG, conjugate residual) solvers, with automatic
  fallback to maintain numerical stability.

All optimizers are subclasses of :class:`torch.optim.Optimizer` for API
compatibility, but they are used with a **closure-based objective** rather than
the standard ``loss.backward()`` + ``optimizer.step()`` pattern.

Where to start?
---------------

- **New to the library?** Start with the :doc:`Quickstart <quickstart>` tutorial
  for a step-by-step introduction.

- **Looking for a specific optimizer?** Browse the
  :doc:`API reference <api_reference>` for a complete list of available
  optimizers, their parameters, and usage notes.

- **Want to build a custom optimizer?** See the
  :doc:`Custom optimizers <api_reference.building_algorithms>` guide to assemble
  your own from curvature estimators, line-search solvers, and trust-region
  solvers.

- **Need the full class and function reference?** Explore the auto-generated
  :doc:`Module reference <auto/torch_numopt>` for detailed API documentation of
  every component.

Quick example
-------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch_numopt import SupervisedLearningObjective, GaussNewtonLS

   model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
   loss_fn = nn.MSELoss()

   X = torch.randn(100, 10)
   y = torch.sum(X, dim=1, keepdim=True)

   optimizer = GaussNewtonLS(model.parameters(), lr_init=1.0)
   objective = SupervisedLearningObjective(model, loss_fn, optimizer)
   objective.set_data(X, y)

   for epoch in range(100):
       optimizer.step(objective)
       with torch.no_grad():
           loss = objective.loss(*objective.params)
           print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

Documentation contents
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Quickstart <quickstart>
   API reference <api_reference>
   Module reference <auto/torch_numopt>

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`