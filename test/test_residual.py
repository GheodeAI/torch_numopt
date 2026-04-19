import torch
import pytest
from torch_numopt.residual import GaussNewton, GaussNewtonLS, LevenbergMarquardt, LevenbergMarquardtLS

@pytest.mark.parametrize("opt_class", [GaussNewton, GaussNewtonLS])
def test_gauss_newton(model_and_data, opt_class):
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
    for _ in range(2):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss

@pytest.mark.parametrize("opt_class", [LevenbergMarquardt, LevenbergMarquardtLS])
def test_levenberg_marquardt(model_and_data, opt_class):
    model, X, y, loss_fn = model_and_data
    if "LS" in opt_class.__name__:
        optimizer = opt_class(
            model,
            lr_init=1.0,
            lr_method="keep",
            mu=0.01,
            mu_dec=0.5,
            mu_max=10.0,
            fletcher=False,
            line_search_method="backtrack",
            line_search_cond="armijo",
            max_iter=5,
        )
    else:
        optimizer = opt_class(
            model,
            lr_init=1.0,
            lr_method="keep",
            mu=0.01,
            mu_dec=0.5,
            mu_max=10.0,
            fletcher=False,
        )
    
    initial_loss = loss_fn(model(X), y).item()
    for _ in range(2):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
        # For LM, we need to call update after step (as per implementation)
        with torch.no_grad():
            pred = model(X)
            new_loss = loss_fn(pred, y)
        optimizer.update(new_loss)
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss