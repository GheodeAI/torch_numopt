# Torch Numerical Optimization (torch_numopt)

**torch_numopt** is a PyTorch library that provides a comprehensive collection of **classical numerical optimization algorithms** for machine learning and general‑purpose differentiable optimization. While PyTorch’s built‑in optimizers focus on first‑order methods (SGD, Adam, etc.), this package implements second‑order and quasi‑Newton methods that can significantly accelerate convergence on problems where curvature information is available or can be approximated.

The library is designed to be modular, extensible, and efficient. It offers:

- A wide range of optimizers – Newton (exact Hessian), Gauss‑Newton, Levenberg‑Marquardt, nonlinear Conjugate Gradient, L‑BFGS, AdaHessian, and gradient descent with line search or trust region.
- Flexible curvature estimators – exact Hessian (full or block‑diagonal), Gauss‑Newton approximation (full or block), diagonal Hessian via Hutchinson’s method, and identity (for first‑order methods).
- Line search and trust‑region frameworks – backtracking, interpolation, bisection, Cauchy point, dogleg, exact (with Lagrange multiplier), and Steihaug‑Toint CG.
- Linear solvers – direct (Cholesky, LU, pseudo‑inverse, least‑squares) and iterative (CG, truncated CG, conjugate residual) solvers, with automatic fallback to maintain numerical stability.

All optimizers are subclasses of `torch.optim.Optimizer` for API compatibility, but **they are not used in the standard PyTorch workflow** (i.e., you do not call `loss.backward()` and then `optimizer.step()` separately). Instead, they accept an **objective function** (a closure) that computes the loss and automatically performs backpropagation. This design is inspired by the closure mechanism in PyTorch’s own LBFGS optimizer and allows the optimizers to evaluate the objective multiple times (e.g., during line search or trust‑region steps) without redundant computations.

The library also provides a `SupervisedLearningObjective` class that wraps a model, loss function, and data loader. **Crucially, the objective is stateless with respect to the data – you must call `set_data(X, y)` before each call to `optimizer.step()` to specify which data the loss should be computed on.** This allows you to switch between different batches or the full dataset at each iteration.

## Important: Deterministic vs. Stochastic Problems

The methods in this library are designed for **deterministic** optimization, where the objective function and its gradient are computed exactly from the full dataset (or a fixed batch) without random subsampling. They rely on stable curvature estimates, which are **not reliable** when the objective is noisy (e.g., mini‑batch training with high variance). Attempting to use these optimizers directly on stochastic problems will often lead to poor performance or divergence.

For stochastic settings, we recommend using PyTorch’s native optimizers (SGD, Adam, etc.) or combining these methods with a **full‑batch** evaluation.

## When to use these optimizers

Second‑order methods (Newton, Gauss‑Newton) require computing or approximating the Hessian matrix, which can be **memory‑intensive** for large models. The full Hessian scales quadratically with the number of parameters, making it impractical for modern deep learning architectures with millions of parameters.

However, the package also includes **Newton‑CG** (inexact Newton with conjugate gradients) and **truncated Newton** variants that never form the full Hessian; they only compute Hessian‑vector products. This makes them applicable to moderately large models. For very large models, diagonal approximations (AdaHessian) or quasi‑Newton (L‑BFGS) are more appropriate.

In practice, these methods are best suited for:
- Small‑ to medium‑sized neural networks (e.g., a few hidden layers).
- Problems where the loss is smooth and convex (or nearly so) – e.g., nonlinear regression, physics‑informed neural networks, and classic optimization benchmarks.
- Scenarios where faster convergence (fewer iterations) is more important than per‑iteration cost.

## Installation

```bash
pip install torch-numopt
```

or install from source:

```bash
git clone https://github.com/GheodeAI/torch_numopt.git
cd torch_numopt
pip install -e .
```


## Usage Example

The following snippet demonstrates a simple supervised learning problem using the Gauss‑Newton optimizer with line search.

```python
import torch
import torch.nn as nn
from torch_numopt import SupervisedLearningObjective, GaussNewtonLS

model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
loss_fn = nn.MSELoss()

X = torch.randn(100, 10)
y = torch.sum(X, dim=1, keepdim=True)

optimizer = GaussNewtonLS(model.parameters(), lr_init=1.0, line_search_method="backtrack")
objective = SupervisedLearningObjective(model, loss_fn, optimizer)
# Set the data at least once before the first step. The loss will be computed
# on this data during optimizer.step().
objective.set_data(X, y)

for epoch in range(100):
    # If you are using mini-batches, you would call objective.set_data(batch_X, batch_y)
    # here to switch to a new batch before each step.
    
    optimizer.step(objective)  # objective.closure() is called internally
    
    with torch.no_grad():
        loss = objective.loss(*objective.params)
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")
```

> **Note**: This is **not** the typical PyTorch pattern of `loss.backward()` + `optimizer.step()`. The optimizer takes full control of evaluation and backpropagation, which is necessary for line‑search and trust‑region methods that re‑evaluate the objective multiple times per step as well as the curvature calculation.

If you prefer to use a standard PyTorch optimizer (e.g., SGD) with the same objective, that is also possible: the `ObjectiveFunction` can be used as a closure for `torch.optim.LBFGS` or any other optimizer that accepts a closure.

For more detailed examples and advanced usage (trust‑region, custom curvature estimators, etc.), please refer to the [documentation](https://torch-numopt.readthedocs.io/).

## Available Optimizers

| Optimizer               | Curvature Used         | Step Selection       | Notes |
|-------------------------|------------------------|----------------------|-------|
| `Newton`                | Exact Hessian          | Fixed learning rate  | Full or block‑diagonal |
| `NewtonLS`              | Exact Hessian          | Line search          | Recommended for Newton |
| `NewtonTR`              | Exact Hessian          | Trust region         | Robust; uses exact or CG solver |
| `NewtonCG` / `NewtonCGLS` | Exact Hessian (Hvp)  | Fixed / line search  | Inexact Newton, memory‑efficient |
| `NewtonCGTR`            | Exact Hessian (Hvp)    | Trust region (Steihaug‑Toint) | Inexact Newton with trust region; memory‑efficient |
| `GaussNewton`           | JᵀJ approximation      | Fixed learning rate  | For least‑squares |
| `GaussNewtonLS`         | JᵀJ approximation      | Line search          | – |
| `GaussNewtonTR`         | JᵀJ approximation      | Trust region         | – |
| `LevenbergMarquardt`    | Damped JᵀJ             | Adaptive damping (μ) | Trust‑region damping |
| `InexactLevenbergMarquardt`| Damped JᵀJ             | Adaptive damping (μ) | Trust‑region damping, Inexact, memory-efficient solver|
| `ConjugateGradient`     | None (uses gradient)   | Fixed learning rate  | Non‑linear CG (FR, PR, etc.) |
| `ConjugateGradientLS`   | None (uses gradient)   | Line search          | – |
| `LBFGS`                 | Approx. inverse Hessian | Fixed learning rate  | Quasi‑Newton, memory‑efficient |
| `LBFGSLS`               | Approx. inverse Hessian | Line search          | Recommended for L‑BFGS |
| `AdaHessian`            | Diagonal Hessian (Hutchinson) | Fixed learning rate | Adaptive, scalable, uses momentum |
| `AdaHessianLS`          | Diagonal Hessian (Hutchinson) | Line search | Adaptive, scalable, uses momentum |
| `DiagonalNewton`        | Diagonal Hessian (Hutchinson) | Fixed learning rate | – |
| `DiagonalNewtonLS`      | Diagonal Hessian (Hutchinson) | Line search | – |
| `GradientDescent`       | Identity               | Fixed learning rate  | Baseline; requires manual tuning |
| `GradientDescentLS`     | Identity               | Line search          | Robust step‑size selection |
| `GradientDescentTR`     | Identity               | Trust region (Cauchy point) | Simple trust‑region baseline |
| `GradientDescentLipschitz` | Identity            | Lipschitz‑estimated LR | Adaptive learning rate, no tuning required |

## Curvature Estimators

The library provides several ways to obtain curvature information:

- `ExactHessianCalculator` – full Hessian matrix (via `torch.func.hessian`).
- `ExactBlockHessianCalculator` – block‑diagonal Hessian (one block per parameter tensor).
- `GaussNewtonApproximation` – JᵀJ (full) for least‑squares.
- `GaussNewtonBlockApproximation` – block‑diagonal JᵀJ.
- `HutchinsonDiagonalApproximation` – diagonal Hessian via random Rademacher vectors.
- `NaiveIdentityCalculator` – identity (no curvature).

All estimators support damping (identity or Fletcher) and can be combined with any optimizer.

## Linear Solvers

The package includes a flexible `solve_system` module that can handle:

- Direct solvers: `"solve"` (LU), `"cholesky"`, `"pinv"`, `"pinv-trunc"`, `"lsqrs"`, `"safe-lsqrs"`.
- Iterative solvers: `"cg"`, `"cg-trunc"`, `"cr"`.

The system automatically falls back to a more stable solver when numerical issues are detected.

## References

- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
- Tan, H. H., & Lin, K. H. (2019). Review of second‑order optimization techniques in artificial neural networks backpropagation. *IOP Conference Series: Materials Science and Engineering*, 495, 012003.
- Martens, J. (2010). Deep learning via Hessian‑free optimization. *ICML*.

## Acknowledgements

This package draws inspiration from the [torchimize](https://github.com/hahnec/torchimize) library. We thank the authors of that project, in particular for their Gauss-Newton and Levenberg-Marquardt implementation.

## License

This project is licensed under the **GNU Lesser General Public License v3.0** – see the [LICENSE](LICENSE) file for details.