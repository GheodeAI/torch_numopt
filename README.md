# Torch Numerical Optimization

This package implements classic numerical optimization methods for training Artificial Neural Networks. 

These methods are not very common in Deep learning frameworks due to their computational requirements, like in the case of Newton-Raphson and Levemberg-Marquardt, which require a large amount of memory since they use information about the second derivative of the loss function. For this reason, it is recommended that these algorithms are applied only to Neural Networks with few hidden layers.

There are also a couple of methods that do not require that much memory such as SGD with line search and the Conjugate Gradient method.

## References
[Numerical Optimization, Jorge Nocedal, Stephen J. Wright](https://link.springer.com/book/10.1007/978-0-387-40065-5)

[Review of second-order optimization techniques in artificial neural networks backpropagation, Hong Hui Tan, Kim Hann Lin](https://iopscience.iop.org/article/10.1088/1757-899X/495/1/012003/meta)

Note: Approximate Greatest Descent is not interesting enough to be included, the author of the method is shared with the author of the review paper, making it's inclusion in the review seem biased. The method can be replicated by applying damping to the hessian on Newton's method along with a trust region method to calculate $\mu$.

## Planned optimizers

- [x] Newton-Raphson
- [x] Gauss-Newton
- [x] Levenberg-Marquard (LM)
- [x] Stochastic Gradient Descent with Line Search
- [x] Conjugate Gradient
- [x] AdaHessian
- [ ] Quasi-Newton (LBFGS already in pytorch)
- [ ] Hessian-free / truncated Newton

If you feel like there's a missing algorithm you can open an issue with the name of the algorithm with some references and a justification why you think it should be included in the package.
