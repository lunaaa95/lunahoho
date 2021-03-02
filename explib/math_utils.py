"""Math Utilities."""
import numpy as np
from scipy import linalg
from sklearn.metrics import confusion_matrix as sklearn_confusion_mat
import numbers


def check_random_state(seed):
    """
    [sklearn]
    Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def logspace(start, stop, num, endpoint=True):
    """
    Return numbers spaced evenly on a log scale.
    :param start: the starting value
    :param stop: the final value
    :param num: number of points to generate
    :param endpoint: whether to include the stop point
    :return: arr
    """
    a = np.log10(start)
    b = np.log10(stop)
    arr = np.logspace(a, b, num, endpoint=endpoint)
    return arr


def fast_logdet(A):
    """
    [sklearn]
    Compute log(det(A)) for A symmetric.
    :param A: a symmetric matrix or 2d array
    :return: ld
    """
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld


def log_likelihood(emp_cov, precision):
    """
    [sklearn]
    Computes the sample mean of the log_likelihood under a covariance model.
    :param emp_cov: empirical covariance matrix
    :param precision: precision matrix
    :return: log_likelihood_
    """
    p = precision.shape[0]
    log_likelihood_ = - np.sum(emp_cov * precision) + fast_logdet(precision)
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.
    return log_likelihood_


def max_lambda(emp_cov):
    """
    [sklearn]
    Find the maximum lambda for which there are some non-zeros off-diagonal.
    :param emp_cov: covariance matrix
    :return: the maximum lambda
    """
    A = np.copy(emp_cov)
    A.flat[::A.shape[0] + 1] = 0
    return np.max(np.abs(A))


def glasso_objective(cov_, precision_, lambda_, diagonal_penalty=False):
    """
    Evaluate the graphical lasso objective function.
    :param cov_: covariance matrix
    :param precision_: precision matrix
    :param lambda_: penalty parameter or matrix
    :param diagonal_penalty: whether to apply penalty to diagonal elements
    :return: cost
    """
    p = precision_.shape[0]
    if isinstance(lambda_, np.ndarray):
        Lambda = lambda_
    else:
        Lambda = np.full((p, p), lambda_)
    if not diagonal_penalty:
        np.fill_diagonal(Lambda, 0.)
    cost = - 2. * log_likelihood(cov_, precision_) - p * np.log(2 * np.pi)
    cost += (Lambda * np.abs(precision_)).sum()
    return cost


def glasso_dual_gap(cov_, precision_, lambda_, diagonal_penalty=False):
    """
    Evaluate the graphical lasso dual gap.
    :param cov_: covariance matrix
    :param precision_: precision matrix
    :param lambda_: penalty parameter or matrix
    :param diagonal_penalty: whether to apply penalty to diagonal elements
    :return: dual gap
    """
    p = precision_.shape[0]
    if isinstance(lambda_, np.ndarray):
        Lambda = lambda_
    else:
        Lambda = np.full((p, p), lambda_)
    if not diagonal_penalty:
        np.fill_diagonal(Lambda, 0.)
    gap = (cov_ * precision_).sum()
    gap += (Lambda * np.abs(precision_)).sum()
    gap -= p
    return gap


def soft_threshold(x, threshold):
    """
    Soft threshold a scalar or an array by given threshold.
    :param x: a scalar or an array
    :param threshold: a scalar or an array
    :return: soft-thresholded result
    """
    if isinstance(x, numbers.Real):  # scalar
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.
    elif isinstance(x, np.ndarray):  # array
        pos_mask = x > threshold
        neg_mask = x < -threshold
        ret = np.zeros_like(x)
        ret[pos_mask] = (x - threshold)[pos_mask]
        ret[neg_mask] = (x + threshold)[neg_mask]
        return ret
    else:
        raise ValueError('Unsupported data type: {}'.format(type(x)))


def count_edges(prec_mat, tol=1e-8):
    """
    Count the number of edges in a precision matrix.
    :param prec_mat: a precision matrix
    :param tol: tolerance
    :return: nb of edges
    """
    mask = ~np.isclose(np.triu(prec_mat, k=1), 0., atol=tol)
    return mask.sum()


def confusion_matrix(true_prec, estimated_prec, tol=1e-8):
    """
    Generate the confusion matrix.
    :param true_prec: the true precision matrix
    :param estimated_prec: the estimated precision matrix
    :param tol: tolerance
    :return: conf_mat
    """
    idx = np.triu_indices_from(true_prec, k=1)
    true_label = ~np.isclose(true_prec[idx], 0., atol=tol)
    estimated_label = ~np.isclose(estimated_prec[idx], 0., atol=tol)
    conf_mat = sklearn_confusion_mat(true_label, estimated_label)
    return conf_mat


def metrics_from_confusion(conf_mat):
    """
    Generate metrics from a confusion matrix.
    :param conf_mat: a confusion matrix
    :return: accuracy, precision, recall, f1
    """
    tn, fp, fn, tp = conf_mat.astype(float).ravel()
    accuracy = (tn + tp) / (conf_mat.sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2. / (1. / precision + 1. / recall)
    return accuracy, precision, recall, f1
