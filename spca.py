import numpy as np

from numba import njit
from scipy.sparse.linalg import eigsh


@njit
def tpower_get_leading(A, x, card, max_iter, tol):
    k = 0
    while k < max_iter:
        y = A.dot(x) # a step of power iteration
        y /= np.linalg.norm(y)

        abs_y = np.abs(y)
        c = np.sort(abs_y)[card]
        y[abs_y < c] = 0. # keep top n-card elements
        y /= np.linalg.norm(y) # renormalize

        if np.sum(np.square(y - x)) < tol: # converge
            return y

        x = y.copy()
        k += 1

    # not converge
    print(f'Maximal iteration achieved: {max_iter}')
    return x


def tpower(A, n, s, t, /, m=3, max_iter=500, eps=1e-5):
    """
    :param A: kernel matrix (sample covariance matrix)
    :param n: number of samples (number of features)
    :param s: number of (sparse) components
    :param t: a list of cardinalities in (0, 1) or a single number
    :param m: number of reduction in warmup phase
    :param max_iter: maximal iterations for extracting each component
    :param eps: convergence criteria
    :return: s sparse eigenvectors
    """
    sp_U = np.zeros((n, s))
    sumU = np.zeros((n, n))
    A2 = np.linalg.matrix_power(A, 2)
    err = np.zeros((s,))

    if type(t) is float: t = [t for _ in range(s)]

    for i in range(s):
        _, x_init = eigsh(A, k=1, which='LM')
        x_init = x_init.flatten()

        if t[i] < 1:
            # multistage warm-up
            card = n
            rf = t[i] ** (1/m)
            for _ in range(m):
                card = int(np.floor(rf * card))
                x_init = tpower_get_leading(A, x_init, n - card, max_iter, eps)

        x = x_init.flatten()
        sumU += np.outer(x, x)
        err[i] = np.sum(A2 * np.linalg.matrix_power(np.eye(n) - sumU, 2))
        sp_U[:, i] = x
        A = deflation(A, x)
    return sp_U, A, err


def MML0(A, n, s, rho, /, max_iter=500, eps=1e-5):
    """
    :param A: kernel matrix (sample covariance matrix)
    :param n: number of samples (number of features)
    :param s: number of (sparse) components
    :param rho: a list of penalized coefficients or a single number
    :param max_iter: maximal iterations for extracting each component
    :param eps: convergence criteria
    :return: s sparse eigenvectors
    """
    sp_U = np.zeros((n, s))
    sumU = np.zeros((n, n))
    A2 = np.linalg.matrix_power(A, 2)
    err = np.zeros((s,))
    if type(rho) is float: rho = [rho for _ in range(s)]

    for i in range(s):
        C = np.max(np.diag(A))
        _, x_init = eigsh(A, k=1, which='LM')
        x_init = x_init.flatten()

        if rho[i] < 1:
            x_init = mml0_get_leading(A, x_init, rho[i]*C, max_iter, eps)

        x = x_init.flatten()
        sumU += np.outer(x, x)
        err[i] = np.sum(A2 * np.linalg.matrix_power(np.eye(n) - sumU, 2))
        sp_U[:, i] = x
        A = deflation(A, x)
    return sp_U, A, err


@njit
def mml0_get_leading(A, x, rho, max_iter, tol):
    k = 0
    p = x.shape[0]
    y = x.copy()
    while k < max_iter:
        a = 2*A.dot(y) # a step of power iteration
        idx = np.argsort(-np.abs(a))
        idx1 = np.argsort(idx)
        a = a[idx]

        if abs(a[0]) <= rho:
            y = np.zeros(p)
            y[0] = np.sign(a[0])

        else:
            a2 = np.square(a)
            ssum = a2[0]
            s = 1
            while (ssum + a2[s]) ** 0.5 > (ssum ** 0.5 + rho):
                ssum += a2[s]
                s += 1
            y[:s] = a[:s] / ssum ** 0.5
            y[s:] = 0.

        y = y[idx1]
        if np.sum(np.square(y - x)) < tol: # converge
            return y
        x = y.copy()
        k += 1

    # not converge
    print(f'Maximal iteration achieved: {max_iter}')
    return x


@njit
def deflation(A, x):
    """ Schur complement deflation """
    Ax = A.dot(x)
    A = A - np.outer(Ax, Ax) / np.sum(x * Ax)
    return A