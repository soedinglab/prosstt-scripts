# coding: utf-8

"""
collection of functions that make notebooks for various simulation proofs
less cluttered
"""
import sys

import numpy as np
import pandas as pd

import scipy as sp

from scipy.special import gammaln
from scipy.special import psi as psi0
from scipy.special import gamma as Gamma

from scipy.stats import lognorm
from scipy.stats import norm

from scipy.integrate import trapz


def print_progress(iteration, total, prefix='', suffix='', decimals=1):
    """
    Call in a loop to create a terminal-friendly text progress bar. Contributed
    by Greenstick on stackoverflow.com/questions/3173320.

    Parameters
    ----------
        iteration: int
            Current iteration.
        total: int
            Total number of iterations.
        prefix: str, optional
            Prefix string before the progress bar.
        suffix: str, optional
            Suffix string after the progress bar.
        decimals: int, optional
            Positive number of decimals in percent complete.
    """
    bar_length = 80
    format_str = "{0:." + str(decimals) + "f}"
    percent = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    progress_bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, progress_bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


# functions to transform UMIs to real counts (Grün et al, Nature Methods, 2014)
def transcriptize(koi, K):
    K = K * 1.
    m = np.log(1 - koi / K) / np.log(1 - 1 / K)
    return(np.nan_to_num(m))


def transform_df(df, K):
    res = np.zeros(df.shape)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            res[i][j] = transcriptize(df[i][j], K)
    return res


# functions to fit poissons on count data
def poisson(lamb, x):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb**x / Gamma(x + 1)) * np.exp(-lamb)


def log_poisson(lamb, x):
    """poisson pdf, parameter lamb is the fit parameter"""
    return x * np.log(lamb) - gammaln(x + 1) - lamb


def nl_poisson(lamb, x):
    """poisson pdf, parameter lamb is the fit parameter"""
    N = len(x)
    return -np.sum(x * np.log(lamb)) + N * lamb + np.sum(gammaln(x + 1))


def nl_poisson_der(lamb, x):
    """ the negative log-Likelohood-Function"""
    lnl = np.sum(- x / lamb + 1)
    return np.array([lnl])


# functions to fit negative binomials on count data
def negbin(x, theta):
    p, r = theta
    # gamma(x+n) / (gamma(n) * factorial(x)) * p^n * (1-p)^x
    return (Gamma(r + x) * (1 - p)**x * (p)**r /
            (Gamma(r) * sp.special.factorial(x)))


def get_pr(a, b, mu):
    sigma2 = a * mu**2 + b * mu
    p = (sigma2 - mu) / sigma2
    r = mu**2 / (sigma2 - mu)
    return(p, r)


def lognegbin(theta, x):
    p, r = theta
    if isinstance(x, np.ndarray):
        N = len(x)
    else:
        N = 1
    s1 = np.sum(gammaln(x + r))
    s2 = np.sum(gammaln(x + 1))
    s3 = N * gammaln(r)
    s4 = N * np.log(p) * r
    s5 = np.sum(x) * np.log(1 - p)
    res = s1 - s2 - s3 + s4 + s5
    return(-res)


def dlr(theta, x):
    p, r = theta
    N = len(x)
    s1 = np.sum(psi0(x + r))
    s2 = N * psi0(r)
    s3 = N * np.log(p)
    return(-(s1 - s2 + s3))


def dlp(theta, x):
    p, r = theta
    N = len(x)
    s1 = np.sum(x) / (1 - p)
    s2 = N * r / p
#     print(s1, s2)
    return((s1 - s2))


def derivative(theta, x):
    return(np.array([dlp(theta, x), dlr(theta, x)]))


def der_intermed(X, mu, a, b, epsilon=1e-10):
    sigma2 = a * mu**2 + b * mu

    p = (sigma2 - mu) / (sigma2 + epsilon)
    r = mu**2 / (sigma2 - mu + epsilon)
    # p[mu == 0] = 0

    s1 = -r**2 * psi0(X + r)
    s2 = r * X * (1 - p)
    s3 = -r**2 * np.log(1 - p) - r * mu**2 / sigma2
    s5 = -r**2 * psi0(r)

    return s1, s2, s3, s5


def deriv(theta, args):
    theta = deconvolve(theta)
    a, b = theta
    X, mu = args

    s1, s2, s3, s5 = der_intermed(X, mu, a, b)

    res = s1 + s2 + s3 - s5
    da = -np.sum(res, axis=0)
    # dllb is just dlla/mu
    res = res / mu
    db = -np.sum(res, axis=0)

    return np.append(da, db)


def cb(theta):
    a, b = deconvolve(theta)
    print(a[0:10])
    print(b[0:10])
    print("===================================")


def negbin_nll(theta, args, epsilon=1e-10):
    theta = deconvolve(theta)
    a, b = theta
    X, mu = args
    sigma2 = a * mu**2 + b * mu

    p = (sigma2 - mu) / (sigma2 + epsilon)
    r = mu**2 / (sigma2 - mu + epsilon)

    if p == 0 and r == 0:
        return 0
    
    # print(p, r)
    s1 = gammaln(X + r)
    s2 = X * np.log(p)
    s3 = r * np.log(1 - p)
    s4 = gammaln(X + 1)
    s5 = gammaln(r)

    nll = -np.sum(s1 + s2 + s3 - s4 - s5)
    return nll


def deconvolve(theta):
    middle = int(len(theta) / 2)
    a = theta[:middle]
    b = theta[middle:]
    return(np.array([a, b]))


# fit functions
def fit_negbins(G, X, step=100, verbose=False):
    N = X.shape[0]
    if G > X.shape[1]:
        G = X.shape[1]

    negbin_pvals = np.zeros(G)
    a_res = np.zeros(G)
    b_res = np.zeros(G)
    nb_success = np.repeat(False, G)

    for g in range(G):
        xx = X[:, g]
        theta = np.array([0.3, 2])

        bnds = ((0, None), (1, None))
        mu = np.mean(xx)
        args = list([xx, mu])
        opt_negbin = sp.optimize.minimize(negbin_nll, theta, args=(args), jac=deriv,
                                          method="L-BFGS-B", bounds=bnds)
        a_res[g], b_res[g] = opt_negbin.x
        nb_success[g] = opt_negbin.success
        # save the BIC for the negbin fit
        if opt_negbin.success:
            negbin_pvals[g] = - opt_negbin.fun - 0.5 * 2 * np.log(N)
        else:
            likelihood = - negbin_nll(opt_negbin.x, args)
            negbin_pvals[g] = likelihood - 0.5 * 2 * np.log(N)
        if verbose:
            print_progress(g, G)
    return negbin_pvals, a_res, b_res, nb_success


def fit_poissons(G, X, step=100, verbose=False):
    N = X.shape[0]
    if G > X.shape[1]:
        G = X.shape[1]

    pois_pvals = np.zeros(G)
    lambda_res = np.zeros(G)
    pois_success = np.repeat(False, G)

    for g in range(G):
        xx = X[:, g]
        # initial parameter estimates
        lamb_init = np.mean(xx)

        # poisson fit
        opt_pois = sp.optimize.minimize(nl_poisson, (lamb_init), args=(xx),
                                        jac=nl_poisson_der, method="L-BFGS-B")
        lambda_res[g] = opt_pois.x
        pois_success[g] = opt_pois.success
        pois_pvals[g] = - opt_pois.fun[0] - 0.5 * 1 * np.log(N)
        if verbose:
            print_progress(g, G)
    return pois_pvals, lambda_res, pois_success


def fit_normals(G, X, step=100, verbose=False):
    N = X.shape[0]
    if G > X.shape[1]:
        G = X.shape[1]

    norm_pvals = np.zeros(G)
    mu_res = np.zeros(G)
    s2_res = np.zeros(G)
    s = np.zeros(G)

    for g in range(G):
        xx = np.log(X[:, g] + 1)

        mu_res[g], s2_res[g] = sp.stats.norm.fit(data=xx)
        mynorm = sp.stats.norm(loc=mu_res[g], scale=s2_res[g])
        prob = mynorm.pdf(xx)
        total_prob = sum(prob)
        prob = prob / total_prob
        norm_pvals[g] = np.sum(np.log(prob)) - 0.5 * 2 * np.log(N)
        if verbose:
            print_progress(g, G)
    return norm_pvals, mu_res, s2_res


def save_all(folder, prefix, norm_all, pois_all, negbin_all):
    norm_pvals, mu_res, s2_res = norm_all
    pois_pvals, lambda_res, pois_success = pois_all
    negbin_pvals, a_res, b_res, nb_success = negbin_all

    np.savetxt(folder + "/" + prefix + "_norm_pvals.txt", norm_pvals)
    np.savetxt(folder + "/" + prefix + "_norm_mu.txt", mu_res)
    np.savetxt(folder + "/" + prefix + "_norm_s2.txt", s2_res)
    # np.savetxt(folder + "/" + prefix + "_norm_s.txt", s_res)
    np.savetxt(folder + "/" + prefix + "_pois_pvals.txt", pois_pvals)
    np.savetxt(folder + "/" + prefix + "_pois_lambda.txt", lambda_res)
    np.savetxt(folder + "/" + prefix + "_pois_success.txt", pois_success)
    np.savetxt(folder + "/" + prefix + "_negbin_pvals.txt", negbin_pvals)
    np.savetxt(folder + "/" + prefix + "_negbin_a.txt", a_res)
    np.savetxt(folder + "/" + prefix + "_negbin_b.txt", b_res)
    np.savetxt(folder + "/" + prefix + "_negbin_success.txt", nb_success)


def hist_and_fit(data, x, ax, label, lwd=2):
    mu, std = norm.fit(data)
    ax.hist(data, bins=50, alpha=0.4, label=label, normed=True)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=lwd)
    ax.set_title(label)
    return(mu, std)


def read_all(folder, prefix, sign_thresh=2):
    negbin = np.loadtxt(folder + "/" + prefix + "_negbin_pvals.txt")
    norm = np.loadtxt(folder + "/" + prefix + "_norm_pvals.txt")
    pois = np.loadtxt(folder + "/" + prefix + "_pois_pvals.txt")

    res = {}
    pvals = np.matrix([negbin, pois, norm])
    diffs = np.apply_along_axis(differences, 0, pvals)

    best_negbin = (diffs[0] > sign_thresh) & (diffs[1] > sign_thresh)
    best_poisson = (diffs[0] < -sign_thresh) & (diffs[2] > sign_thresh)
    best_lognorm = (diffs[1] < -sign_thresh) & (diffs[2] < -sign_thresh)

    delta_BIC_above_cutoff = np.array([sum(best_negbin),
                                       sum(best_poisson),
                                       sum(best_lognorm)])
    total_genes_surveyed = np.array([len(norm), len(pois), len(negbin)])
    return delta_BIC_above_cutoff, total_genes_surveyed


def successful(folder, prefix, epsilon=1e-6, sign_thresh=2):
    read = np.loadtxt(folder + "/" + prefix + "_negbin_success.txt")
    alphas = np.log(np.loadtxt(
        folder + "/" + prefix + "_negbin_a.txt") + epsilon)
    betas = np.log(np.loadtxt(folder + "/" + prefix +
                              "_negbin_b.txt") + epsilon - 1)
    keep = np.array(read, dtype=bool) & (alphas > -6)
    # res = {}
    # pvals = np.matrix([negbin, pois, norm])
    # diffs = np.apply_along_axis(differences, 0, pvals)

    # keep = (diffs[0] > sign_thresh) & (diffs[1] > sign_thresh)# & (alphas > np.exp(epsilon))
    return alphas[keep], betas[keep]


def differences(x):
    """
    calculates the outer product of differences between all elements in x
    and returns the upper right triangular matrix in flat form.
    """
    a = np.array(x).flatten()
    diff = np.subtract.outer(a, a)
    iu1 = np.triu_indices(3, k=1)
    return diff[iu1]


def plot_fits(X, g, a_res, b_res, lambda_res, mu_res, s2_res):
    # goodness of fit
    xx = X[:, g]
    fig, ax = plt.subplots(nrows=2)
    x = np.sort(xx)
    ecdf = ECDF(xx)
    ax[0].plot(x, ecdf.y[1:], label="ECDF")
    p, r = get_pr(a_res[g], b_res[g], np.mean(xx))
    x = np.arange(min(x), max(x), 0.1)
    yy = np.real(np.exp(cm.lognegbin(x, [p, r])))
    y = np.cumsum(yy)
    y /= y[-1]
    ax[0].plot(x, y, "k--", label="negbin")
    yy = np.exp(log_poisson(lambda_res[g], x))
    y = np.cumsum(yy)
    y /= y[-1]
    ax[0].plot(x, y, label="poisson")
    yy = sp.stats.norm.cdf(np.log(x + 1), loc=mu_res[g], scale=s2_res[g])
    ax[0].plot(x, yy, label="lognorm")
    ax[0].legend()

    ax[1].hist(xx, normed=True)
    y = np.real(np.exp(cm.lognegbin(x, [p, r])))
    ax[1].plot(x, y, "k--", label="negbin")
    y = np.exp(log_poisson(lambda_res[g], x))
    ax[1].plot(x, y, label="poisson")
    lognorm = sp.stats.norm(loc=mu_res[g], scale=s2_res[g])
    y = lognorm.pdf(np.log(x + 1))
    ax[1].plot(x, y, label="lognorm")
    plot.show()
