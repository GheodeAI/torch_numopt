import torch
import pytest
from torch_numopt.second_order import Newton, NewtonLS, AdaHessian, AdaHessianLS

@pytest.mark.parametrize("opt_class", [Newton, NewtonLS])
def test_newton_optimizer(model_and_data, opt_class):
    model, X, y, loss_fn = model_and_data
    optimizer = opt_class(
        model,
        lr_init=1.0,
        lr_method="keep",
        damping="identity",
        mu=0.1,
    )
    # For NewtonLS we need to pass line_search args if required; adjust based on signature
    # We'll use conditional instantiation
    if "LS" in opt_class.__name__:
        optimizer = opt_class(
            model,
            lr_init=1.0,
            lr_method="keep",
            damping="identity",
            mu=0.1,
            line_search_method="backtrack",
            line_search_cond="armijo",
            max_iter=5,
        )
    else:
        optimizer = opt_class(model, lr_init=1.0, lr_method="keep", damping="identity", mu=0.1)
    
    initial_loss = loss_fn(model(X), y).item()
    for _ in range(2):  # Hessian computation is heavy, keep steps minimal
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss

@pytest.mark.parametrize("opt_class", [AdaHessian, AdaHessianLS])
def test_adahessian_optimizer(model_and_data, opt_class):
    model, X, y, loss_fn = model_and_data
    if "LS" in opt_class.__name__:
        optimizer = opt_class(
            model,
            lr_init=1.0,
            lr_method="keep",
            line_search_method="backtrack",
            line_search_cond="armijo",
            max_iter=5,
        )
    else:
        optimizer = opt_class(model, lr_init=1.0, lr_method="keep")
    
    initial_loss = loss_fn(model(X), y).item()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss