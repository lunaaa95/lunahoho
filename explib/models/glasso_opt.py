import numpy as np
from scipy import linalg
import networkx as nx
import operator

from explib.math_utils import soft_threshold, logspace, count_edges, max_lambda
from .py_quic import quic

import logging

logger = logging.getLogger(__name__)


def glasso(emp_cov, lambda_, rho=1.0, max_it=200, tol=1e-7, diagonal_penalty=False):
    """
    Spare covariance matrix estimation using ADMM.
    Parameters
    ----------
    emp_cov : np.ndarray
        the empirical covariance matrix
    lambda_ : float or np.ndarray
        the penalty coefficient, a scalar or a matrix
    rho : float
        the multiplier of ADMM
    max_it : int
         maximum iteration
    tol : float
        tolerance of convergence
    diagonal_penalty : bool
        whether to apply penalty to diagonal elements
    Returns
    -------
    Z : np.ndarray
        the precision matrix
    residuals : list [float]
        the primal residuals of each iteration
    flag : bool
        whether the algorithm converges
    """
    p = emp_cov.shape[0]
    if isinstance(lambda_, np.ndarray):
        Lambda = lambda_
    else:
        Lambda = np.full((p, p), lambda_)
    if not diagonal_penalty:
        np.fill_diagonal(Lambda, 0)
    # X = np.identity(p, dtype=float)  # precision
    Z = np.identity(p, dtype=float)  # auxiliary
    U = np.zeros((p, p), dtype=float)  # dual variable
    residuals = list()
    flag = True
    for it in range(max_it):
        # update X
        d, Q = linalg.eigh(rho * (Z - U) - emp_cov)
        new_d = (d + np.sqrt(d * d + 4. * rho)) / 2. / rho
        X = (new_d * Q) @ Q.T
        # update Z
        Z = soft_threshold(X + U, Lambda / rho)
        # update U
        R = X - Z   # primal residual
        U += R
        r = linalg.norm(R, 'fro')
        residuals.append(r)
        if r < tol:
            break
    else:
        print('Reach max iteration!')
        flag = False
    residuals = np.array(residuals)
    return Z, residuals, flag


def glasso_with_screening(emp_cov, lambda_, rho=1.0, max_it=200, tol=1e-7, diagonal_penalty=False):
    """
    Sparse covariance matrix estimation with screening rule.
    Parameters
    ----------
    emp_cov : np.ndarray
        the empirical covariance matrix
    lambda_ : float or np.ndarray
        the penalty coefficient, a scalar or a matrix
    rho : float
        the multiplier of ADMM
    max_it : int
         maximum iteration
    tol : float
        tolerance of convergence
    diagonal_penalty : bool
        whether to apply penalty to diagonal elements
    Returns
    -------
    prec_ : np.ndarray
        the precision matrix
    total_flag : bool
        whether the algorithm converges
    """
    # transform scalar to matrix
    p = emp_cov.shape[0]
    if isinstance(lambda_, np.ndarray):
        Lambda = lambda_
    else:
        Lambda = np.full((p, p), lambda_)
    if not diagonal_penalty:
        np.fill_diagonal(Lambda, 0)
    mask = np.abs(emp_cov) > Lambda
    graph = nx.from_numpy_matrix(mask)  # build graph
    prec_ = np.zeros_like(emp_cov)
    total_flag = True
    for components in nx.connected_components(graph):
        idx = np.array(list(components))  # components is a set
        if len(idx) == 1:
            i = idx[0]
            prec_[i, i] = 1. / (emp_cov[i, i] + Lambda[i, i])
        else:
            sub_cov = emp_cov[np.ix_(idx, idx)]
            sub_lambda = Lambda[np.ix_(idx, idx)]
            sub_prec, residuls, flag = glasso(sub_cov, sub_lambda, rho, max_it, tol, diagonal_penalty)
            prec_[np.ix_(idx, idx)] = sub_prec
            total_flag &= flag
    return prec_, total_flag


def quic_with_screening(emp_cov, lambda_, max_it=200, tol=1e-7, diagonal_penalty=False):
    """
    Sparse covariance matrix estimation with screening rule.
    Parameters
    ----------
    emp_cov : np.ndarray
        the empirical covariance matrix
    lambda_ : float or np.ndarray
        the penalty coefficient, a scalar or a matrix
    max_it : int
         maximum iteration
    tol : float
        tolerance of convergence
    diagonal_penalty : bool
        whether to apply penalty to diagonal elements
    Returns
    -------
    prec_ : np.ndarray
        the precision matrix
    """
    # transform scalar to matrix
    p = emp_cov.shape[0]
    if isinstance(lambda_, np.ndarray):
        Lambda = lambda_
    else:
        Lambda = np.full((p, p), lambda_)
    if not diagonal_penalty:
        np.fill_diagonal(Lambda, 0)
    mask = np.abs(emp_cov) > Lambda
    graph = nx.from_numpy_matrix(mask)  # build graph
    prec_ = np.zeros_like(emp_cov)
    for components in nx.connected_components(graph):
        idx = np.array(list(components))  # components is a set
        if len(idx) == 1:
            i = idx[0]
            prec_[i, i] = 1. / (emp_cov[i, i] + Lambda[i, i])
        else:
            sub_cov = emp_cov[np.ix_(idx, idx)]
            sub_lambda = Lambda[np.ix_(idx, idx)]
            sub_prec, *_ = quic(sub_cov, sub_lambda, tol=tol, max_iter=max_it)
            prec_[np.ix_(idx, idx)] = sub_prec
    return prec_


def glasso_search(emp_cov, target_edges, ratio=.01,
                  n_refinements=5, n_points=4, tol=1e-08,
                  with_screening=True, **kwargs):
    """
    Search the best lambda that makes the number of edges close to the ground truth.
    Parameters
    ----------
    emp_cov : np.ndarray
        the empirical covariance matrix
    target_edges : int
        the number of edges in the ground truth
    ratio : float
        the initial minimum lambda is ratio * max_lambda
    n_refinements : int
        the number of refinements to do
    n_points : int
        the number of points on the interval in each refinement
    tol : the tolerance for np.isclose
    with_screening : bool
        whether to use screening rule
    kwargs : dict
        parameters for the glasso solver
    Returns
    -------
    best_index : int
    path : list
        list of tuples: (lambda_, edges, flag, prec_mat)
    """
    if n_points < 4:
        raise ValueError("'n_points' should be at least 4.")
    path = list()  # tuple: (lambda, edges, flag, prec_mat)
    # select solver
    if with_screening:
        solver = glasso_with_screening
    else:
        solver = glasso
    # Determine the interval for search
    lambda_1 = max_lambda(emp_cov)  # max lambda
    lambda_0 = lambda_1 * ratio
    # cache
    searched_lambdas = set()
    for i_refine in range(n_refinements):
        lambda_list = logspace(lambda_0, lambda_1, n_points)[
                      ::-1]  # large->small
        for lambda_ in lambda_list:
            if lambda_ in searched_lambdas:
                continue
            searched_lambdas.add(lambda_)
            prec_, *_, flag = solver(emp_cov, lambda_, **kwargs)
            n_edges = count_edges(prec_, tol)
            path.append((lambda_, n_edges, flag, prec_))
        # Sort path
        path = sorted(path, key=operator.itemgetter(0), reverse=True)
        # Find the best index
        scores = np.array([p[1] for p in path]) - target_edges
        best_index = np.argmin(np.abs(scores))
        # Refine the grid
        if scores[best_index] == 0:
            break
        if best_index == 0:  # the maximum lambda
            lambda_1 = path[0][0]
            lambda_0 = path[1][0]
        elif best_index == len(path) - 1:  # the minimum lambda
            lambda_1 = path[best_index][0]
            lambda_0 = lambda_1 * ratio
        else:
            lambda_1 = path[best_index - 1][0]
            lambda_0 = path[best_index + 1][0]

    path = sorted(path, key=operator.itemgetter(0), reverse=True)
    return best_index, path
