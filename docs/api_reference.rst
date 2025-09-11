API reference
=============

Base Optimization classes
-------------------------

.. csv-table::
   :header: "Class", "Description"

   ":py:class:`~torch_numopt.CustomOptimizer`", "Base optimization class. Optimization with inputs x, y, and loss function"
   ":py:class:`~torch_numopt.LineSearchOptimizer`", "Class implementing line search methods for optimizaton."
   ":py:class:`~torch_numopt.SecondOrderOptimizer`", "Class implementing methods for calculating and formatting the hessian matrix for optimization."

Available algorithms
--------------------

.. csv-table::
   :header: "Algorithm", "Description"

   ":py:class:`~torch_numopt.GradientDescentLS`", "Vanilla gradient descent with line search."
   ":py:class:`~torch_numopt.ConjugateGradientLS`", "Conjugate gradient descent with line search"
   ":py:class:`~torch_numopt.NewtonLS`", "Newton's method for optimization with line search"
   ":py:class:`~torch_numopt.GaussNewtonLS`", "Gauss-Newton algorithm with line search."
   ":py:class:`~torch_numopt.LevenbergMarquardtLS`", "Levenberg-Marquardt algorithm with line search."
   ":py:class:`~torch_numopt.AdaHessian`", "AdaHessian algorithm."
