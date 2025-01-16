import numpy as np
import matplotlib.pyplot as plt


def check_float(potential_float):
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


def fitSVD(x_i, y_i, sigma_i, M):
    N = len(x_i)
    X_ij = np.array([x_i[j] ** i for i in range(M) for j in range(N)]).reshape((M, N)).T
    A_ij = (X_ij.T / sigma_i).T
    b_i = y_i / sigma_i
    u, s, vt = np.linalg.svd(A_ij, full_matrices=False)
    sigma = np.zeros((A_ij.shape[0], A_ij.shape[1]))
    for i in range(min(A_ij.shape[0], A_ij.shape[1])):
        sigma[i, i] = s[i]
    a_i = np.zeros((M))
    for i in range(M):
        a_i += (np.dot(u[:, i], b_i) / s[i]) * vt[i, :]
    Cov_ij = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            for i in range(M):
                Cov_ij[j, k] += (vt.T[j, i] * vt.T[k, i]) / (s[i] ** 2)
    y_fit_i = (a_i * X_ij).sum(axis=1)
    ChiSq = (((y_i - y_fit_i) / sigma_i) ** 2.0).sum() / (N - M)
    return a_i, Cov_ij, y_fit_i, ChiSq


def plot_extrap(x, y, y_err, inds, col, xlim=0):
    a_i, Cov_ij, y_fit_i, ChiSq = fitSVD(x[inds], y[inds], y_err[inds], 2)
    if xlim == 0:
        xlim = x[inds].max()
    x_extr = np.linspace(0, xlim, 3)
    y_extr = (x_extr * a_i[1]) + a_i[0]
    y0 = a_i[0]
    y0_err = Cov_ij[0, 0] ** 0.5
    line2 = plt.errorbar(
        0,
        y0,
        y0_err,
        color=col,
        marker="^",
        capsize=7,
    )
    (line3,) = plt.plot(x_extr, y_extr, color=col, ls="--")
    print("ChiSq", ChiSq)
    print("Cov_ij", Cov_ij)
    return y0, y0_err
