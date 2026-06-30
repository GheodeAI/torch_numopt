import pytest
import torch
from torch_numopt.utils.param_operations import *


@pytest.mark.parametrize(
    "params_a,params_b,params_sum",
    [
        ((torch.tensor([1, 2]),), (torch.tensor([3, 4]),), (torch.tensor([4, 6]),)),
        ((torch.tensor([1, 2]), torch.tensor([3, 4])), (torch.tensor([3, 4]), torch.tensor([5, 6])), (torch.tensor([4, 6]), torch.tensor([8, 10]))),
        (
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])),
            (torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9, 10, 11])),
            (torch.tensor([5, 7, 9]), torch.tensor([11, 13, 15, 17, 19])),
        ),
    ],
)
def test_param_sum(params_a, params_b, params_sum):
    result = param_add(params_a, params_b)
    assert all(torch.all(res_p == expect_p) for res_p, expect_p in zip(result, params_sum))


@pytest.mark.parametrize(
    "params_a,params_b,params_sum",
    [
        ((torch.tensor([3, 4]),), (torch.tensor([1, 2]),), (torch.tensor([2, 2]),)),
        ((torch.tensor([3, 4]), torch.tensor([5, 6])), (torch.tensor([1, 2]), torch.tensor([3, 4])), (torch.tensor([2, 2]), torch.tensor([2, 2]))),
        (
            (torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9, 10, 11])),
            (torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7, 8])),
            (torch.tensor([3, 3, 3]), torch.tensor([3, 3, 3, 3, 3])),
        ),
    ],
)
def test_param_sub(params_a, params_b, params_sum):
    result = param_diff(params_a, params_b)
    assert all(torch.all(res_p == expect_p) for res_p, expect_p in zip(result, params_sum))


@pytest.mark.parametrize(
    "params,expected",
    [
        ((torch.randn(3, 4),), (torch.Size([3, 4]),)),
        (
            (torch.randn(2, 3), torch.randn(5)),
            (torch.Size([2, 3]), torch.Size([5])),
        ),
        ((), ()),  # empty tuple
    ],
)
def test_param_sizes(params, expected):
    assert param_sizes(params) == expected


@pytest.mark.parametrize(
    "params",
    [
        (torch.ones(2, 3),),
        (torch.ones(5), torch.ones(3, 1)),
    ],
)
def test_param_zero_like(params):
    result = param_zero_like(params)
    assert all(r.shape == p.shape for r, p in zip(result, params))
    assert all(torch.all(r == 0.0) for r in result)


@pytest.mark.parametrize(
    "flat_vec,shapes,expected_shapes",
    [
        (torch.arange(6, dtype=torch.float32), [(2, 3)], [(2, 3)]),
        (torch.arange(6, dtype=torch.float32), [(3, 2)], [(3, 2)]),
        (
            torch.arange(8, dtype=torch.float32),
            [(2, 2), (4,)],
            [(2, 2), (4,)],
        ),
    ],
)
def test_param_reshape_like(flat_vec, shapes, expected_shapes):
    # Create dummy params with target shapes
    dummy = tuple(torch.empty(s) for s in shapes)
    result = param_reshape_like(flat_vec, dummy)
    assert all(r.shape == es for r, es in zip(result, expected_shapes))
    # Check that concatenated result equals original flat vector
    assert torch.equal(
        torch.cat([r.flatten() for r in result]),
        flat_vec,
    )


def test_param_reshape_like_size_mismatch():
    """Should raise or fail; important for safety."""
    flat = torch.arange(5)
    params = (torch.empty(2, 3),)  # 6 elements
    with pytest.raises((AssertionError, RuntimeError)):
        param_reshape_like(flat, params)


@pytest.mark.parametrize(
    "params,expected",
    [
        ((torch.tensor([[1, 2], [3, 4]]),), torch.tensor([1.0, 2.0, 3.0, 4.0])),
        (
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])),
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        ),
        ((), torch.tensor([])),  # edge case
    ],
)
def test_param_flatten(params, expected):
    result = param_flatten(params)
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "params,expected",
    [
        ((torch.tensor([3.0, 4.0]),), 5.0),
        ((torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0])), torch.sqrt(torch.tensor(13.0))),
        ((torch.zeros(5),), 0.0),
    ],
)
def test_param_norm(params, expected):
    expected_val = expected.item() if isinstance(expected, torch.Tensor) else expected
    assert torch.isclose(param_norm(params), torch.tensor(expected_val))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((torch.tensor([1.0, 2.0]),), (torch.tensor([3.0, 4.0]),), 11.0),
        (
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
            1 * 5 + 2 * 6 + 3 * 7 + 4 * 8,  # 5+12+21+32=70
        ),
    ],
)
def test_param_dot(a, b, expected):
    assert torch.isclose(param_dot(a, b), torch.tensor(expected, dtype=a[0].dtype))


@pytest.mark.parametrize(
    "scalar,params,expected",
    [
        (2.0, (torch.tensor([1.0, 2.0]),), (torch.tensor([2.0, 4.0]),)),
        (
            -1.0,
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -4.0])),
        ),
    ],
)
def test_param_scalar_prod(scalar, params, expected):
    result = param_scalar_prod(scalar, params)
    assert all(torch.equal(r, e) for r, e in zip(result, expected))


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((torch.tensor([2.0, 3.0]),), (torch.tensor([4.0, 5.0]),), (torch.tensor([8.0, 15.0]),)),
        (
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
            (torch.tensor([5.0, 12.0]), torch.tensor([21.0, 32.0])),
        ),
    ],
)
def test_param_mult(a, b, expected):
    result = param_mult(a, b)
    assert all(torch.equal(r, e) for r, e in zip(result, expected))


@pytest.mark.parametrize(
    "a,b,scale,expected",
    [
        (
            (torch.tensor([1.0, 2.0]),),
            (torch.tensor([3.0, 4.0]),),
            2.0,
            (torch.tensor([1.0 + 6, 2.0 + 8]),),  # [7,10]
        ),
        (
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([5.0, 6.0]), torch.tensor([7.0, 8.0])),
            0.5,
            (torch.tensor([1.0 + 2.5, 2.0 + 3.0]), torch.tensor([3.0 + 3.5, 4.0 + 4.0])),
        ),
    ],
)
def test_param_scaled_add(a, b, scale, expected):
    result = param_scaled_add(a, b, scale)
    assert all(torch.allclose(r, e) for r, e in zip(result, expected))


def test_param_transpose_2d():
    p = (torch.tensor([[1, 2], [3, 4]]),)
    result = param_transpose(p)
    assert result[0].shape == (2, 2)
    assert torch.equal(result[0], torch.tensor([[1, 3], [2, 4]]))


def test_param_transpose_multiple():
    p = (
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([[5, 6]]),  # shape (1,2) -> (2,1)
    )
    result = param_transpose(p)
    assert torch.equal(result[0], torch.tensor([[1, 3], [2, 4]]))
    assert torch.equal(result[1], torch.tensor([[5], [6]]))


@pytest.mark.parametrize(
    "params,expected",
    [
        ((torch.tensor([1.0, -2.0]),), (torch.tensor([-1.0, 2.0]),)),
        (
            (torch.tensor([0.0, 1.0]), torch.tensor([-3.0, 4.0])),
            (torch.tensor([0.0, -1.0]), torch.tensor([3.0, -4.0])),
        ),
    ],
)
def test_param_neg(params, expected):
    result = param_neg(params)
    assert all(torch.equal(r, e) for r, e in zip(result, expected))


def test_param_copy():
    p = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0]))
    c = param_copy(p)
    # Check values equal
    assert all(torch.equal(a, b) for a, b in zip(p, c))
    # Check they are not the same object (clone)
    c[0][0] = 999.0
    assert p[0][0] == 1.0  # original unchanged


@pytest.mark.parametrize(
    "params,expected",
    [
        ((torch.tensor([1.0, 2.0]),), True),
        ((torch.tensor([1.0, float("nan")]),), False),
        ((torch.tensor([float("inf")]), torch.tensor([0.0])), False),
        ((torch.tensor([]),), True),  # edge: empty tensor all() returns True
    ],
)
def test_param_is_finite(params, expected):
    assert param_is_finite(params) == expected
