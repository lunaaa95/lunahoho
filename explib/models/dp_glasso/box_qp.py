import numpy as np


def solve_box_qp(Q, b, rho, u0=None, max_it=100, tol=1e-7):
    n = Q.shape[0]
    rho = np.zeros(n) + rho
    if u0 is None:
        u = (np.random.rand(n) - .5) * 2 * rho
    else:
        u = u0.copy()

    grad = Q @ (u + b)
    fval_0 = (grad * (u + b)).sum()
    for it in range(max_it):
        for i in range(n):
            t = Q[i, i] * u[i]
            u_star = -(grad[i] - t) / Q[i, i]
            if abs(u_star) > rho[i]:
                u_star = np.sign(u_star) * rho[i]
            if u_star != u[i]:
                grad += (u_star - u[i]) * Q[i]
                u[i] = u_star
        fval = (grad * (u + b)).sum()
        if abs(fval - fval_0) / abs(fval_0 + 1e-7) < tol:
            break
        fval_0 = fval
    else:
        print('BOXQP Reach Max Iteration!')
    return u, it, fval
