import numpy as np

from explib.math_utils import check_random_state
from explib.math_utils import fast_logdet
import sys


def _get_svd():
    if sys.platform == 'darwin':
        def f(X):
            def _f(A):
                s, v = np.linalg.eigh(A.T @ A)
                s = s[s > 0]
                s = np.sqrt(s)
                v = v[:, -len(s):]
                u = A @ v / s
                return u, s, v.T
            m, n = X.shape
            if m < n:
                v, s, uh = _f(X.T)
                return uh.T, s, v.T
            else:
                u, s, vh = _f(X)
                return u, s, vh
    else:
        def f(A):
            u, s, vh = np.linalg.svd(A, full_matrices=False)
            return u, s, vh
    return f


svd = _get_svd()


def solve_cubic(b, c):
    """
    Solve the cubic equation(s):
    x^3 + bx^2 + cx + d = 0,
    where b <= 0 and 0 <= c <= 1.5.
    (In this case, the equation has a unique real root.)
    Parameters
    ----------
    b : float or np.ndarray
        the coefficients of the quadratic and constant term
    c : float or np.ndarray
        the coefficient of the linear term
    Returns
    -------
    r : float or np.ndarray
        the real root of the cubic equation(s)
    """
    assert np.alltrue((b <= 0) & (c <= 1.5) & (c >= 0)), f'\n=========\n{b}\n{c}'
    p = b * b - 3. * c
    q = b * (2. * b * b - 9. * c + 27.)
    t = np.sqrt(q * q - 4. * p ** 3)
    u1 = np.cbrt((q + t) / 2.)
    u2 = np.cbrt((q - t) / 2.)
    r = -(b + u1 + u2) / 3.
    return r


def solve_neg_logdet(A, beta):
    """
    solver for the following problem:
    min -logdet(I + Z^T Z) + \frac{\beta}{2} ||Z - A||_F^2
    Parameters
    ----------
    A : np.ndarray
        
    beta : float
        a positive number
    Returns
    -------

    """
    # u, sigmas, vh = np.linalg.svd(A, full_matrices=False)
    # u, sigmas, vh = slinalg.svd(A.T, full_matrices=False)
    u, sigmas, vh = svd(A)
    sigmas = solve_cubic(-sigmas, 1 - 2. / beta)
    Z = (sigmas * u) @ vh
    return Z


def solve_multi_projected_logdet(A_list, S_list, d,
                                 rho=2., gamma=1.05, beta=1.,
                                 max_it=100, tol=1e-6, seed=0):
    """
    min \sum{ -logdet(I + A_i^T X^T X A_i) + Tr(S_i X^T X) }
    Parameters
    ----------
    A_list : list [np.ndarray]
        a list of A
    S_list : list [np.ndarray]
        a list of S
    d : int
        the latent dimension
    rho : float
        the multiplier (> 2)
    gamma : float
        the increasing factor (> 1)
    beta : float
        the l2 penalty coefficent
    max_it : int
        the maximum possible iteration
    tol : float
        the stop criterion
    seed : int
        the seed of random generators

    Returns
    -------
    X : np.ndarray
        the optimal X
    residuals : list [float]
        the residuals
    """
    rng = check_random_state(seed)
    m = A_list[0].shape[0]
    # initial guess
    X = rng.rand(d, m)
    # auxiliary vars Z_i = X @ A_i
    # dual variables Y_i
    Y_list = [np.zeros((d, A.shape[1])) for A in A_list]
    AAT_list = [A @ A.T for A in A_list]
    r_list = []
    for it in range(max_it):
        # update Z
        Z_list = [solve_neg_logdet(X @ A - Y / rho, rho) for A, Y in zip(A_list, Y_list)]
        # update X, solve XC = D
        C = np.sum([rho * AAT + 2. * S for AAT, S in zip(AAT_list, S_list)], axis=0)
        C += np.identity(m) * beta
        D = np.sum([(rho * Z + Y) @ A.T for A, Z, Y in zip(A_list, Z_list, Y_list)], axis=0)
        # use pseudo-inverse because C may not be full-rank
        X = D @ np.linalg.pinv(C)
        # update Y
        R_list = [Z - X @ A for Z, A in zip(Z_list, A_list)]
        for Y, R in zip(Y_list, R_list):
            Y += rho * R
        # update rho
        rho *= gamma
        # log
        r = sum(np.linalg.norm(R, 'fro') for R in R_list)
        r_list.append(r)
        if r < tol:
            break
    return X, np.array(r_list)


def calc_multi_logdet_obj(A_list, S_list, X):
    """
    Calculate \sum{ -logdet(I + A_i^T X^T X A_i) + Tr(S_i X^T X) }
    Parameters
    ----------
    A_list : list [np.ndarray]
    S_list : list [np.ndarray]
    X : np.ndarray

    Returns
    -------
    fval : float
    """
    XTX = X.T @ X
    fval = 0.
    for A, S in zip(A_list, S_list):
        m, p = A.shape
        d, m = X.shape
        v = fast_logdet(np.identity(p) + A.T @ XTX @ A)
        fval += -v + np.trace(XTX @ S)
    return fval
