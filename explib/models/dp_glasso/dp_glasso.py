import numpy as np
from .box_qp import solve_box_qp


def solve_dp_glasso(S, rho, X=None, invX=None, max_it=100, tol=1e-5, rtol=1e-7):
    p = S.shape[0]
    if X is None:
        X = np.diag(1. / (np.diag(S) + rho))
    else:
        X = X.copy()
    if invX is None:
        U = np.diag(np.full(p, rho))
        invX = S + U
    else:
        U = invX - S
    full_idx = np.arange(p)
    oldX = X.copy()
    for it in range(max_it):
        for i in range(p):
            idx = np.delete(full_idx, i)
            subX = X[np.ix_(idx, idx)]
            colS = S[idx, i]
            colU = U[idx, i]

            u, *_ = solve_box_qp(subX, colS, rho, u0=colU, max_it=1000, tol=1e-7)
            # grad of the above QP
            x_hat = -subX @ (u + colS) / (S[i, i] + rho)
            # zero out
            x_hat[np.abs(u) < rho * (1 - rtol)] = 0.  # grad=0 if u in the interior of feasible domain
            x_hat[np.abs(x_hat) < rtol] = 0.
            # update X
            X[idx, i] = x_hat
            X[i, idx] = x_hat
            X[i, i] = (1. - (x_hat * (u + colS)).sum()) / (S[i, i] + rho)
            # update U
            U[idx, i] = u
            U[i, idx] = u
            U[i, i] = rho
        err = np.abs(oldX - X).max() / np.abs(oldX).max()
        print(err)
        if err < tol:
            break
    else:
        print('DPGLASSO Reach Max Iteration')
    invX = S + U
    return X, invX
