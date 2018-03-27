# coding: utf-8

"""
collection of functions that are needed for a, b fitting
"""

import numpy as np

def func(theta, x):
    alpha, beta = theta
    mu, sigma = x
    negbin_var = alpha * mu**2 + beta*mu
    res = np.sqrt(np.sum((sigma - negbin_var)**2))
    return res

def deriv(theta, x):
    alpha, beta = theta
    mu, sigma = x
    negbin_var = alpha * mu**2 + beta * mu
    function = func(theta, x)

    partial_a = np.sum(- mu**2 * (sigma - negbin_var)) / function
    partial_b = np.sum(- mu * (sigma - negbin_var)) / function
    res = [partial_a, partial_b]
    return res


def read_negbin_params(folder, group):
    p = np.loadtxt(folder + "/" + group + "_negbin_p.txt")
    r = np.loadtxt(folder + "/" + group + "_negbin_r.txt")
    success = np.loadtxt(folder + "/" + group +
                         "_negbin_success.txt").astype(bool)
    which = np.loadtxt(folder + "/" + group + "_negbin_which.txt").astype(bool)
    return p, r, success, which