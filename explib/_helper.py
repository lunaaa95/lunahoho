"""Some helper functions for developing."""
import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .math_utils import check_random_state


__all__ = ['make_synthetic_matrix', 'imshow', 'plot_table', 'show_tensor']


def make_synthetic_matrix(n_features, n_samples, sparsity=.98, random_state=0):
    """
    Make synthetic precision matrix and empirical covariance matrix.
    :param n_features: number of features (nodes)
    :param n_samples: number of samples used to generate empirical cov mat
    :param sparsity: a float in (0, 1), the larger, the sparser
    :param random_state: random state for reproducibility
    :return: emp_cov, prec
    """
    prng = check_random_state(random_state)
    prec = make_sparse_spd_matrix(n_features, alpha=sparsity,
                                  smallest_coef=.4, largest_coef=.7,
                                  random_state=prng)
    cov = linalg.inv(prec)
    d = np.sqrt(np.diag(cov))
    cov /= d
    cov /= d[:, np.newaxis]
    prec *= d
    prec *= d[:, np.newaxis]
    X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    # Estimate the covariance
    emp_cov = np.dot(X.T, X) / n_samples
    return emp_cov, prec


def plot_table(mat, width=.15, ratio=4):
    """
    Plot a matrix in a table.
    :param mat: a matrix that will be presented
    :param width: column width
    :param ratio: height ratio
    :return: fig
    """
    vmax = np.abs(mat).max()
    vals = np.around(mat, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
    table = plt.table(cellText=vals, colWidths=[width]*vals.shape[1],
                      loc='center', cellColours=plt.cm.RdBu_r(
                        Normalize(-vmax, vmax)(mat)))
    table.scale(1, ratio)
    return fig


def imshow(mat, vmax=None, vmin=None, bar=True, tol=1e-8, *, gray=False):
    """
    Plot a matrix.
    :param mat: a 2d np.ndarray
    :param vmax: the maximum value in mat
    :param vmin: the minimum value in mat
    :param bar: whether to show the colorbar
    :param gray: whether to show the gray image
    :return: None
    """
    if gray:
        plt.imshow(np.isclose(mat, 0., atol=tol),
                   interpolation='nearest', cmap='gray')
    else:
        if vmax is None:
            vmax = np.abs(mat).max()
        if vmin is None:
            vmin = -vmax
        plt.imshow(np.ma.masked_equal(mat, 0),
                   interpolation='nearest', vmin=vmin, vmax=vmax,
                   cmap=plt.cm.RdBu_r)
        if bar:
            plt.colorbar()


def show_tensor(list_or_tensor, gray=True):
    """
    Show a tensor or a list of matrices.
    :param list_or_tensor: a tensor of a list of matrices
    :param gray: gray or colorful
    :return: fig
    """
    n = len(list_or_tensor)
    fig = plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(list_or_tensor[i], bar=False, gray=gray)
    return fig
