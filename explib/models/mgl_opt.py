import numpy as np

from explib.math_utils import check_random_state, glasso_objective, fast_logdet
from .logdet_opt import solve_multi_projected_logdet
from .glasso_opt import quic_with_screening as solve_glasso

# lambda_2=1. -> magl ; lambda_2=0. ->glasso 
# S is the sample covariance matrix.
# Ak is the i-th variable’s attributes in the k-th task
def solve_mgl(S_list, A_list, lambda_1=.1, lambda_2=0., lambda_3=1.,
              beta=.1,
              d=None, outer_max_it=100, outer_tol=1e-6,
              inner_max_it=100, inner_tol=1e-6,
              diagonal_penalty=False,
              rho_logdet=2., gamma=1.05,
              seed=0):
    rng = check_random_state(seed)
    _, m = A_list[0].shape
    p_list = [A.shape[0] for A in A_list]
    U = rng.rand(m, d)
    # U = np.zeros((m, d))
    scaled_A_list = [A.T / np.sqrt(beta) for A in A_list]  # !!! Transpose

    fvals = []
    for it in range(outer_max_it):
        # update Precision matrices
        UUT = U @ U.T
        S_prime_list = [S + lambda_2 * (beta * np.identity(p) + A @ UUT @ A.T)
                        for p, S, A in zip(p_list, S_list, A_list)]
        X_list = [solve_glasso(S_prime / (1. + lambda_2),
                               lambda_1 / (1. + lambda_2),
                               max_it=inner_max_it,
                               tol=inner_tol,
                               diagonal_penalty=diagonal_penalty)
                  for S_prime in S_prime_list]
        # update Projection matrix
        U, _ = solve_multi_projected_logdet(scaled_A_list,
                                            S_list=[A.T @ X @ A
                                                    for A, X in zip(A_list, X_list)],
                                            d=d, rho=rho_logdet,
                                            beta=lambda_3 / lambda_2,
                                            gamma=gamma, max_it=inner_max_it,
                                            tol=inner_tol, seed=seed)
        U = U.T
        fval = calc_mgl_obj(S_list, A_list, X_list, U,
                            lambda_1, lambda_2, lambda_3, beta, diagonal_penalty)
        print(fval)
        fvals.append(fval)
        if it == 0:
            continue
        last_fval = fvals[-2]
        if abs(fval - last_fval) / abs(last_fval) < outer_tol:
            break
    else:
        print('Reach Max Iteration!')
    return X_list, U, fvals


def calc_mgl_obj(S_list, A_list, X_list, U,
                 lambda_1, lambda_2, lambda_3, beta, diagonal_penalty=False):
    glasso_fval = 0.
    logdet_fval = 0.
    # Glasso
    for S, A ,X in zip(S_list, A_list, X_list):
        glasso_fval += glasso_objective(S, X, lambda_1, diagonal_penalty)
        K = beta * np.identity(A.shape[0]) + A @ U @ U.T @ A.T
        logdet_fval += -fast_logdet(X @ K) + (X * K).sum()
    fval = glasso_fval + lambda_2 * logdet_fval
    fval += lambda_3 * (U ** 2).sum() / 2.
    return fval
