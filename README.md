# Torch Numerical Optimization

This package implements classic numerical optimization methods for training Artificial Neural Networks. 

These methods are not very common in Deep learning frameworks due to their computational requirements, like in the case of Newton-Raphson and Levemberg-Marquardt, which require a large amount of memory since they use information about the second derivative of the loss function. For this reason, it is recommended that these algorithms are applied only to Neural Networks with few hidden layers.

There are also a couple of methods that do not require that much memory such as SGD with line search and the Conjugate Gradient method.

<!-- Implementation of numerical optimization methods for Neural Networks.

Due to computational constraints, methods like Newton-Raphson or Levenberg-Marquardt are to be used with small Neural Networks as they require $O(p^3)$ space for a network with $p$ parameters. -->

## References
[relevant paper](https://iopscience.iop.org/article/10.1088/1757-899X/495/1/012003/pdf)
Approximate Greatest Descent is a waste of time, the author of the method is shared with the author of the review paper. It is also a bit badly written. It's just a Newton's method with damping on the hessian and a trust region method to calculate $\mu$, works well, but it's not very interesting.


## Planned optimizers

- [x] Newton-Raphson
- [x] Gauss-Newton
- [x] Levenberg-Marquard (LM)
- [x] ~~Approximate Greatest Descent (AGD)~~
- [x] Stochastic Gradient Descent with Line Search
- [x] Conjugate Gradient
- [ ] Quasi-Newton (LBFGS already in pytorch)
- [ ] Hessian-free / truncated Newton
