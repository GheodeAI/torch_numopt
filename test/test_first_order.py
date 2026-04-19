import torch
import pytest
from torch_numopt.first_order import GradientDescentLS, ConjugateGradient, ConjugateGradientLS

@pytest.mark.parametrize("opt_class", [GradientDescentLS, ConjugateGradientLS])
def test_first_order_ls_optimizer(model_and_data, opt_class):
    model, X, y, loss_fn = model_and_data
    optimizer = opt_class(
        model,
        lr_init=1.0,
        lr_method="scaled",
        line_search_method="backtrack",
        line_search_cond="armijo",
        max_iter=5,
    )
    
    initial_loss = loss_fn(model(X), y).item()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
    
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss

def test_conjugate_gradient_non_ls(model_and_data):
    model, X, y, loss_fn = model_and_data
    optimizer = ConjugateGradient(model, lr_init=0.01, lr_method="keep", cg_method="PRP+")
    initial_loss = loss_fn(model(X), y).item()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step(X, y, loss_fn)
    final_loss = loss_fn(model(X), y).item()
    assert final_loss < initial_loss