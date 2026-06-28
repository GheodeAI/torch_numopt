import logging
import torch
from .curvature_estimator import CurvatureEstimator
from .objective import ObjectiveFunction
from .utils import (
    fix_cond,
    pinv_svd_trunc,
    param_dot,
    param_scaled_add,
    param_diff,
    param_norm,
    param_flatten,
    param_reshape_like,
    param_zero_like,
)

logger = logging.getLogger(__name__)

direct_solver_set = {"pinv", "pinv-trunc", "solve", "lsqrs", "safe-lsqrs", "cholesky"}
iterative_solver_set = {"cg", "cg-trunc", "cr"}
solver_set = direct_solver_set.union(iterative_solver_set).union({None})


def solve_system(curvature_estimator: CurvatureEstimator, objective: ObjectiveFunction, rhs_params: tuple, solver: str = None, **kwargs) -> tuple:
    # Attempt regular solve
    success = True
    try:
        result = _solve_system(curvature_estimator, objective, rhs_params, solver, **kwargs)
    except torch.linalg.LinAlgError as e:
        success = False
        logger.warning(f"Linear algebra error. {e}")

    if success:
        return result

    # Attempt safe solver
    logger.warning("Fallback to lsqrs solver.")
    success = True
    try:
        result = _solve_system(curvature_estimator, objective, rhs_params, solver="safe-lsqrs", **kwargs)
    except torch.linalg.LinAlgError as e:
        success = False
        logger.warning(f"Linear algebra error. {e}")

    if success:
        return result

    # Fallback to not solving
    logger.error("Cannot solve linear system. Fallback to returning the gradient.")
    return rhs_params


def _solve_system(curvature_estimator: CurvatureEstimator, objective: ObjectiveFunction, rhs_params: tuple, solver: str = None, **kwargs) -> tuple:
    eps = torch.finfo(rhs_params[0].dtype).eps
    params = objective.params

    assert solver in solver_set, f"Solver {solver} not available, use one of {solver_set}."

    if curvature_estimator.ndim == 2:
        if solver in direct_solver_set:
            logger.debug("Using matrix form of curvature, solving linear system...")
            if curvature_estimator.uses_blocks:
                logger.debug("Using block-form matrix...")
                solution_params = [None] * len(rhs_params)
                B_list = curvature_estimator.scaling_matrix(objective, params)
                for i, (rhs, B) in enumerate(zip(rhs_params, B_list)):
                    if not torch.all(torch.isfinite(B)):
                        raise ValueError("NaN found in scaling matrix.")

                    B = fix_cond(B)
                    match solver:
                        case "pinv":
                            h_inv = B.pinverse()
                            p = (h_inv @ rhs.ravel()).reshape(rhs.shape)
                        case "pinv-trunc":
                            h_inv = pinv_svd_trunc(B)
                            p = (h_inv @ rhs.ravel()).reshape(rhs.shape)
                        case "solve":
                            p = torch.linalg.solve(B, rhs.ravel()).reshape(rhs.shape)
                        case "lsqrs":
                            p, *_ = torch.linalg.lstsq(B, rhs.ravel())
                            p = p.reshape(rhs.shape)
                        case "cholesky":
                            L = torch.linalg.cholesky(B)
                            p = torch.cholesky_solve(rhs.ravel()[:, None], L).reshape(rhs.shape)
                        case "safe-lsqrs":
                            p, *_ = torch.linalg.lstsq(B, rhs.ravel(), driver="gels")
                            p = p.reshape(rhs.shape)
                    solution_params[i] = p
            else:
                logger.debug("Using full matrix...")
                rhs = param_flatten(rhs_params)
                B = curvature_estimator.scaling_matrix(objective, params)
                if not torch.all(torch.isfinite(B)):
                    raise ValueError("NaN found in scaling matrix.")

                B = fix_cond(B)
                match solver:
                    case "pinv":
                        h_inv = B.pinverse()
                        p = (h_inv @ rhs.ravel()).reshape(rhs.shape)
                    case "pinv-trunc":
                        h_inv = pinv_svd_trunc(B)
                        p = (h_inv @ rhs.ravel()).reshape(rhs.shape)
                    case "solve":
                        p = torch.linalg.solve(B, rhs.ravel()).reshape(rhs.shape)
                    case "lsqrs":
                        p, *_ = torch.linalg.lstsq(B, rhs.ravel())
                        p = p.reshape(rhs.shape)
                    case "cholesky":
                        L = torch.linalg.cholesky(B)
                        p = torch.cholesky_solve(rhs.ravel()[:, None], L).reshape(rhs.shape)
                    case "safe-lsqrs":
                        p, *_ = torch.linalg.lstsq(B, rhs.ravel(), driver="gels")
                        p = p.reshape(rhs.shape)
                solution_params = param_reshape_like(p, rhs_params)

        elif solver in iterative_solver_set:
            match solver:
                case "cg":
                    solution_params = conjugate_gradient(curvature_estimator, objective, rhs_params, **kwargs)
                case "cg-trunc":
                    solution_params = truncated_cg(curvature_estimator, objective, rhs_params, **kwargs)
                case "cr":
                    solution_params = conjugate_residual(curvature_estimator, objective, rhs_params, **kwargs)
        else:
            raise ValueError("A solver must be specified for 2D second order derivatives.")

    elif curvature_estimator.ndim == 1:
        logger.debug("Using diagonal form of curvature, dividing component-wise...")
        B_list = curvature_estimator.scaling_matrix(objective, rhs_params)
        solution_params = tuple(rhs / (h + eps) for rhs, h in zip(rhs_params, B_list))
    elif curvature_estimator.ndim == 0:
        logger.debug("Using scalar form of curvature, dividing by the scalar...")
        B = curvature_estimator.scaling_matrix(objective, rhs_params)
        solution_params = tuple(rhs / (B + eps) for rhs in rhs_params)
    else:
        raise ValueError("Incorrectly dimensioned hessian.")

    return tuple(solution_params)


def conjugate_gradient(curvature_estimator, objective, rhs, max_iter=100, atol=1e-8, tol=1e-4, min_iter=2):
    eps = torch.finfo(rhs[0].dtype).eps

    def damped_hvp(p):
        return param_scaled_add(curvature_estimator.hvp(objective, objective.params, p), p, scale=eps)

    new_params = param_zero_like(rhs)
    res = rhs
    p_dir = res
    rsold = param_dot(res, res)

    effective_tol = max(atol, tol * param_norm(rhs))

    for i in range(max_iter):
        Ap = damped_hvp(p_dir)
        pAp = param_dot(p_dir, Ap)
        alpha = rsold / (pAp + eps)
        new_params = param_scaled_add(new_params, p_dir, scale=alpha)
        res = param_scaled_add(res, Ap, scale=-alpha)
        rsnew = param_dot(res, res)
        if torch.sqrt(rsnew) < effective_tol and i >= min_iter:
            break

        p_dir = param_scaled_add(res, p_dir, scale=rsnew / rsold)
        rsold = rsnew

    return new_params


def truncated_cg(curvature_estimator, objective, rhs, max_iter=100, atol=1e-8, tol=1e-4, min_iter=2):
    eps = torch.finfo(rhs[0].dtype).eps
    params = objective.params

    new_params = param_zero_like(rhs)
    res = rhs
    p_dir = res
    rsold = param_dot(res, res)

    effective_tol = max(atol, tol * param_norm(rhs))

    for i in range(max_iter):
        Ap = curvature_estimator.hvp(objective, params, p_dir)
        pAp = param_dot(p_dir, Ap)
        if pAp <= 0:
            if i == 0:
                new_params = rhs
            break

        alpha = rsold / (pAp + eps)
        new_params = param_scaled_add(new_params, p_dir, scale=alpha)
        res = param_scaled_add(res, Ap, scale=-alpha)
        rsnew = param_dot(res, res)
        if torch.sqrt(rsnew) < effective_tol and i >= min_iter:
            break

        p_dir = param_scaled_add(res, p_dir, scale=rsnew / rsold)
        rsold = rsnew

    return new_params


def conjugate_residual(curvature_estimator, objective, rhs, max_iter=100, atol=1e-8, tol=1e-4, min_iter=2):
    eps = torch.finfo(rhs[0].dtype).eps
    params = objective.params

    next_params = param_zero_like(rhs)
    res = param_diff(rhs, curvature_estimator.hvp(objective, params, next_params))
    p_dir = res
    Ap = curvature_estimator.hvp(objective, params, p_dir)
    Ares = curvature_estimator.hvp(objective, params, res)

    effective_tol = max(atol, tol * param_norm(rhs))

    for i in range(max_iter):
        resAres = param_dot(res, Ares)
        alpha = resAres / (param_dot(Ap, Ap) + eps)
        next_params = param_scaled_add(next_params, p_dir, scale=alpha)
        res = param_scaled_add(res, Ap, scale=-alpha)

        if param_norm(res) < effective_tol and i >= min_iter:
            break

        Ares = curvature_estimator.hvp(objective, params, res)
        beta = param_dot(res, Ares) / (resAres + eps)
        p_dir = param_scaled_add(res, p_dir, scale=beta)
        Ap = param_scaled_add(Ares, Ap, scale=beta)

    return next_params
