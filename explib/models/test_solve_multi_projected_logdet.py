from unittest import TestCase
import numpy as np
from explib.models.logdet_opt import calc_multi_logdet_obj, solve_multi_projected_logdet

class TestSolve_multi_projected_logdet(TestCase):
    def test_solve_multi_projected_logdet(self):
        rng = np.random.RandomState(10)
        with np.errstate(all='raise'):
            for _ in range(10):
                p_list = rng.randint(10, 200, size=(5, ))
                A_list, S_list, X0 = generate_data(p_list, d=20, m=100, seed=0)
                d = max(p_list) + 10
                X, r = solve_multi_projected_logdet(A_list, S_list, d=d, tol=1e-6, max_it=200)
                print(calc_multi_logdet_obj(A_list, S_list, X), len(r), r[-1])


def generate_data(p_list, d, m, seed=0):
    rng = np.random.RandomState(seed)
    A_list = [rng.rand(m, p) for p in p_list]

    X0 = rng.rand(d, m)  # optimal
    XTX = X0.T @ X0

    S_list = []
    for A, p in zip(A_list, p_list):
        S = A @ np.linalg.inv(np.identity(p) + A.T @ XTX @ A) @ A.T
        S_list.append(S)
    print('fval*: {}'.format(calc_multi_logdet_obj(A_list, S_list, X0)))
    return A_list, S_list, X0