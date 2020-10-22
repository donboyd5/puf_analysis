# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 06:56:25 2020

Parallel group_by?
https://pandas.pydata.org/pandas-docs/stable/ecosystem.html?highlight=parallel

https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby

https://stackoverflow.com/questions/1704401/is-there-a-simple-process-based-parallel-map-for-python
https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
https://github.com/ray-project/ray

Hessian vector products:
    https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.29.6143
    https://arxiv.org/pdf/1611.03777.pdf
    https://www.sciencedirect.com/science/article/pii/S0377042708006523
    https://discuss.pytorch.org/t/hessian-vector-product-optimization/40717/4
    https://github.com/scipy/scipy/issues/8644
    https://stats.stackexchange.com/questions/368475/how-the-hessian-matrix-is-used-in-optimization-if-you-cant-invert-it

Large scale least squares (not for this problem but for other uses):
    https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

@author: donbo
"""

# Notes about eclipse


# %% imports
import os
import sys
import requests
import pandas as pd
import numpy as np
import src.microweight as mw
import src.make_test_problems as mtp

import scipy as sp
from scipy.optimize import fmin_slsqp  # no longer needed
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import show_options
from scipy.sparse import lil_matrix
from scipy.optimize import BFGS
from scipy.optimize import SR1

import cvxpy as cp  # probably won't use

import autograd.numpy as np  # stop this as soon as practical as use ag.numpy
from autograd import grad
from autograd import hessian
from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs

import numdifftools as nd


# %% set working directory if not already set
os.getcwd()
os.chdir('C:/programs_python/weighting')
os.getcwd()


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'

HT2_2018 = "18in55cmagi.csv"

# AGI stubs
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]
HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% crosswalks
# This is Peter's xwalk plus mine -- it includes more than we will use
PUFHT2_XWALK = {
    'nret_all': 'N1',  # Total population
    'nret_mars1': 'MARS1',  # Single returns number
    'nret_mars2': 'MARS2',  # Joint returns number
    'c00100': 'A00100',  # AGI amount
    'e00200': 'A00200',  # Salary and wage amount
    'e00200_n': 'N00200',  # Salary and wage number
    'e00300': 'A00300',  # Taxable interest amount
    'e00600': 'A00600',  # Ordinary dividends amount
    'c01000': 'A01000',  # Capital gains amount
    'c01000_n': 'N01000',  # Capital gains number
    # check Social Security
    # e02400 is Total Social Security
    # A02500 is Taxable Social Security benefits amount
    'e02400': 'A02500',  # Social Security total (2400)
    'c04470': 'A04470',  # Itemized deduction amount (0 if standard deduction)
    'c04470_n': 'N04470',  # Itemized deduction number (0 if standard deduction)
    'c17000': 'A17000',  # Medical expenses deducted amount
    'c17000_n': 'N17000',  # Medical expenses deducted number
    'c04800': 'A04800',  # Taxable income amount
    'c04800_n': 'N04800',  # Taxable income number
    'c05800': 'A05800',  # Regular tax before credits amount
    'c05800_n': 'N05800',  # Regular tax before credits amount
    'c09600': 'A09600',  # AMT amount
    'c09600_n': 'N09600',  # AMT number
    'e00700': 'A00700',  # SALT amount
    'e00700_n': 'N00700',  # SALT number
    # check pensions
    # irapentot: IRAs and pensions total e01400 + e01500
    # A01750: Taxable IRA, pensions and annuities amount
    'irapentot': 'A01750',
}
PUFHT2_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
HT2PUF_XWALK = {val: kw for kw, val in PUFHT2_XWALK.items()}
HT2PUF_XWALK
list(HT2PUF_XWALK.keys())


# %% utility functions


def getmem(objects=dir()):
    """Memory used, not including objects starting with '_'.

    Example:  getmem().head(10)
    """
    mb = 1024**2
    mem = {}
    for i in objects:
        if not i.startswith('_'):
            mem[i] = sys.getsizeof(eval(i))
    mem = pd.Series(mem) / mb
    mem = mem.sort_values(ascending=False)
    return mem


# %% program functions

def wsum(grp, sumvars, wtvar):
    """ Returns data frame row with weighted sums of selected variables.

        grp: a dataframe (typically a dataframe group)
        sumvars: the variables for which we want weighted sums
        wtvar:  the weighting variable
    """
    return grp[sumvars].multiply(grp[wtvar], axis=0).sum()


def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)


def targs(x, xmat, targets):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    return np.dot(whs.T, xmat)


def get_diff_weights(geotargets, goal=100):
    """
    difference weights - a weight to be applied to each target in the
      difference function so that it hits its goal
      set the weight to 1 if the target value is zero

    do this in a vectorized way
    """

    # avoid divide by zero warnings
    goalmat = np.full(geotargets.shape, goal)
    with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
        diff_weights = np.where(geotargets != 0, goalmat / geotargets, 1)

    return diff_weights


def f(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    diffs = diffs * diff_weights
    obj = np.square(diffs).sum() * objscale
    return(obj)


def gfun(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    diffs_weighted2 = diffs * np.square(diff_weights)
    grad = 2 * xmat.dot(diffs_weighted2.T)
    return grad.flatten()


# f_hvp = hvp(f)  # maybe I could put this inside of f_hvp_wrap but only call the first time?
def f_hvp_wrap(x, p, xmat, targets, objscale, diff_weights):
    return f_hvp(x, p, xmat=xmat, targets=targets, objscale=objscale, diff_weights=diff_weights)


def hfun(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    # diffs = diffs * diff_weights
    grad = 2 * xmat.dot(diffs.T)
    return grad.flatten()


# minimize(f, x0, jac=jac, hessp=hessp, method='trust-ncg'...)
# also see this: https://justindomke.wordpress.com/2009/01/17/hessian-vector-products/
# approximation:  H(x)v ~ [g(x+rv) - g(x - rv)] / 2r
def hesspfn(x, p, xmat, targets, objscale, diff_weights):
    # approximation for product of Hessian H and an arbitrary vector p
    r = 1e10
    g1 = gfun(x + r*p, xmat, targets, objscale, diff_weights)
    g2 = gfun(x - r*p, xmat, targets, objscale, diff_weights)
    Hp = (g1 + g2) / (2 * r)
    return Hp


def ediag_fn(x, xmat, targets, objscale, diff_weights):
    return ediag_vals  # global!!!


# def hfn2(x, xmat, targets, objscale, diff_weights):
#     return hvals2  # global!!!


# %% automatic differentiation
# define gradient of objective function
# gfn = grad(f)
# hfn = hessian(f)  # from autgrad so we can test vs analytic

# egfn = egrad(f)
# ehfn = egrad(egfn) # no this is not the hessian, it returns a vector

# %% work on hessian function
# it is too big to construct!!!

# hbak = h; sbak = s; xmbak = xmat; dwbak = diff_weights; targbak = targets; whbak = wh  # save
# h = hbak; s = sbak; xmat = xmbak; diff_weights = dwbak; targets = targback; wh = whbak  # restore
p = mtp.Problem(h=7, s=3, k=2)
h = p.h; s = p.s; xmat = p.xmat; targets = p.targets; wh = p.wh
# diff_weights = get_diff_weights(targets)
diff_weights = np.ones(targets.shape)
x0 = np.ones((h, s)) / s
x0 = np.multiply(x0, wh.reshape(x0.shape[0], 1)).flatten()
hfn = hessian(f)
hmat = hfn(x0, xmat, targets, 1, diff_weights)
hms = sp.sparse.coo_matrix(hmat)
hcheck = sp.sparse.find(hms)
hcheck[0].size
hcheck[1].size
hcheck[2].size
h * h * s
hesvals = 2 * np.dot(xmat, xmat.T)  # djb this is the key
hesvals.shape
hesvals
len(np.unique(hesvals))

# hmat is (h *s, h * s)
# inz: should have h * s rows (1 row per person-state combination)
# and s columns -
# get i and j indexes of nonzero values, and data
# inz = np.arange(0, h).repeat(s * h) + 1
# for x in range(0, s)
# get the i indexes for the hessian matrix - there must be a better way
#   first, get repeating values we want
h = int(10e3); s = 50
srepeats = np.tile(np.tile(np.arange(0, s), h), h)
hs_add_factor = (np.arange(0, h) * s).repeat(s * h)
inz = srepeats + hs_add_factor
inz

np.nonzero(hmat)
np.count_nonzero(hmat)

(ivals % s) *
ivals % np.arange(0, h)

[n%3 for n in range(0,21)]
list(range(0, s)) * h
a = np.tile(np.tile(np.arange(0, s), h), h)
a * np.tile(np.arange(0, h), s)
[a + add for add in range(0, h * s, s)]
np.tile(a + 18, h)
ia = list(range(0, s)) * h
ia = np.tile(np.arange(0, s), h)
ia + 3
for()
np.array(ia) + 3
list(np.arange(0, s).repeat(h))
list(np.arange(0, h).repeat(s))
list(np.arange(0, h).repeat([s, 2]))

list(np.arange(0, s))
list(inz)
np.resize(np.arange(s), h)
np.hstack((arr, ) * 3)
np.hstack((arr, ) * 3)
(list(range(0, s)) * h)
jnz = np.arange(0, s * h)
a = np.array([0, 1, 2])
np.tile(np.arange(0, s), h)
np.tile(a, 2)
np.tile(a, (2, 2))


np.tile(a, (2, 1, 2))



hmat2_sparse = sp.sparse.coo_matrix((np.ones(h*s), (inz, jnz)), shape=(h, h * s))
# hmat2_sparse.todense()

# these vals fit into hessian as follows, one hesvec row at a time
# which diag, which cols -- note the repeats
#  0, 0:2x  1, 0:2x  2, 0:2x  3, 0:2x  4, 0:2x  5, 0:2
# -1, 3:5x  0, 3:5x  1, 3:5x  2, 3:5x  3, 3:5x  4, 3:5  # 1st item again
# -2, 6:8x  -1, 6:8x  0, 6:8x  1, 6:8x  2, 6:8x 3, 6:8
# -3, 9:11
# -4, 12:14
# -5, 15:17

# so now we know how to fill the full hessian, later worry
# about just a triangle, then worry about unique values
# each row of hesvec fills as follows:
# row 0: fills cols 0:2, starting rows 0:2, 3:5, 6:8, etc
# row 1:  cols 3:5, same sets of rows
# row5: cols 15:17, same sets of rows
# that will get us the full hessian


# slow approach
hmat_sparse = lil_matrix((h *s, h * s))
hmat_sparse.shape

# look for a faster way to do this next for loop
for valrow in range(0, hesvals.shape[0]):
    if(valrow % 100 == 0):
        print(valrow)
    cols = range(valrow * s, valrow * s + s)
    for valcol in range(0, hesvals.shape[1]):
        rows = range(valcol * s, valcol * s + s)
        hmat_sparse[rows, cols] = hesvals[valrow, valcol]

def hessfn(x, xmat, targets, objscale, diff_weights):
    return hmat_sparse


# %% practice and test on toy problems

# vector product approach
# https://github.com/scipy/scipy/issues/8644

p = mtp.Problem(h=6, s=3, k=2)
p = mtp.Problem(h=500, s=8, k=4)
# p = mtp.Problem(h=1000, s=20, k=10)
# p = mtp.Problem(h=4000, s=20, k=10)
# p = mtp.Problem(h=10000, s=50, k=10)
# p = mtp.Problem(h=30000, s=50, k=20)

# import scipy.sparse as sps

# # A is the matrix
# n = A.shape[0]

# # Define H1
# def H1(x):
#     # Create matvec function
#     def matvec(p):
#         return A.T.dot(A.dot(p))
#     return scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)

# # Define H2
# H2 = BFGS()

# def hessp(x, p):
#     return H1(x).dot(p) + H2(x).dot(p)



xmat = p.xmat
wh = p.wh
targets = p.targets
h = xmat.shape[0]
s = targets.shape[0]
k = targets.shape[1]

diff_weights = get_diff_weights(targets)

A = lil_matrix((h, h * s))
for i in range(0, h):
    A[i, range(i*s, i*s + s)] = 1
A
# b=A.todense()  # ok to look at dense version if small
A = A.tocsr()  # csr format is faster for our calculations
lincon = sp.optimize.LinearConstraint(A, wh, wh)

wsmean = np.mean(wh) / targets.shape[0]
wsmin = np.min(wh) / targets.shape[0]
wsmax = np.max(wh)  # no state can get more than all of the national weight

objscale = 1

bnds = sp.optimize.Bounds(wsmin / 10, wsmax)

# starting values (initial weights), as an array
# x0 = np.full(h * s, 1)
# x0 = np.full(p.h * p.s, wsmean)
# initial weights that satisfy constraints
x0 = np.ones((h, s)) / s
x0 = np.multiply(x0, wh.reshape(x0.shape[0], 1)).flatten()


# verify that starting values satisfy adding-up constraint
np.square(np.round(x0.reshape((h, s)).sum(axis=1) - wh, 2)).sum()

f(x0, xmat, targets, objscale, diff_weights)

diff_weights = np.ones(targets.shape)

# hmat_sparse = sp.sparse.csr_matrix(hmat)


hesvals = 2 * np.dot(xmat, xmat.T)  # djb this is the key
hesvals.shape
hmat_sparse = lil_matrix((h *s, h * s))
hmat_sparse.shape
# look for a faster way to do this next for loop
for valrow in range(0, hesvals.shape[0]):
    if(valrow % 100 == 0):
        print(valrow)
    cols = range(valrow * s, valrow * s + s)
    for valcol in range(0, hesvals.shape[1]):
        rows = range(valcol * s, valcol * s + s)
        hmat_sparse[rows, cols] = hesvals[valrow, valcol]

def hessfn(x, xmat, targets, objscale, diff_weights):
    return hmat_sparse

# coo_matrix((data, (i, j)), [shape=(M, N)])
# to construct from three arrays:
# data[:] the entries of the matrix, in any order
# i[:] the row indices of the matrix entries
# j[:] the column indices of the matrix entries

# hesvals = 2 * np.dot(xmat, xmat.T)  # djb this is the key
# hmat = np.zeros((h * s, h * s))
# # slow fill
# for valrow in range(0, hesvals.shape[0]):
#     cols = range(valrow * s, valrow * s + s)
#     for valcol in range(0, hesvals.shape[1]):
#         rows = range(valcol * s, valcol * s + s)
#         hmat[rows, cols] = hesvals[valrow, valcol]

hessfn(x0, xmat, targets, objscale, diff_weights)
ediag_vals = ehfn(x0, xmat, targets, objscale, diff_weights)

from autograd import hessian_vector_product as hvp

from autograd import grad, jacobian
def hessian(fun, argnum=0):
    return jacobian(jacobian(fun, argnum), argnum)

def make_vjp(fun, argnum=0):
    def vjp_maker(*args, **kwargs):
        start_node, end_node = forward_pass(fun, args, kwargs, argnum)
        if not isnode(end_node) or start_node not in end_node.progenitors:
            warnings.warn("Output seems independent of input.")
            def vjp(g): return start_node.vspace.zeros()
            else:
                def vjp(g): return backward_pass(g, end_node, start_node)
                return vjp, end_node
            return vjp_maker

def make_hvp(fun, argnum=0):
    """Builds a function for evaluating the Hessian-vector product at a point,
    which may be useful when evaluating many Hessian-vector products at the same
    point while caching the results of the forward pass."""
    def hvp_maker(*args, **kwargs):
        return make_vjp(grad(fun, argnum), argnum)(*args, **kwargs)[0]
    return hvp_maker

fnx = make_hvp(f)

def hvp(fun):
    def grad_dot_vector(arg, vector):
        return np.dot(grad(fun)(arg), vector)
    return grad(grad_dot_vector)

def hvp2(fun, xmat, targets, objscale, diff_weights):
    def grad_dot_vector(arg, vector, xmat, targets, objscale, diff_weights):
        return np.dot(grad(fun)(arg), vector)
    return grad(grad_dot_vector)

y = np.random.normal(size=x0.shape)
gdot = lambda u, p1, xmat, targets, objscale, diff_weights: np.dot(gfun(u, xmat, targets, objscale, diff_weights), y)
hess1 = grad(gdot)(x0, p1, xmat, targets, objscale, diff_weights)


from autograd import hessian_vector_product as hvp

hvpfn = hvp2(f)

hvpfn = hvp(f)
import autograd as ad
fnx = ad.hessian_vector_product(f)
p1 = np.ones(x0.size)
hesspfn(x0, p1, xmat, targets, objscale, diff_weights)
hvpfn(x0, p1, xmat, targets, objscale, diff_weights)
fnx(x0, p1, xmat, targets, objscale, diff_weights)


res = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=lincon,  # lincon lincon_feas
               jac=gfun,
               # hess=hessfn, # hessfn '2-point',  # 2-point 3-point
               # hess='2-point',
               # hess=ediag_fn,
               # hessp=hesspfn,
               hessp=hvpfn,
               args=(xmat, targets, 1, diff_weights),
               options={'maxiter': 100, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})  # default AugmentedSystem NormalEquationc

# 2-point |  22   |  18   |  142  | +1.7339e-10 | 2.22e+05 | 5.95e-05 | 2.27e-13 |
# `gtol` termination condition is satisfied.
# Number of iterations: 22, function evaluations: 18, CG iterations: 142, optimality: 5.95e-05, constraint violation: 2.27e-13, execution time: 1.3e+01 s.

# analytic dense |  14   |  10   |  63   | +6.9278e-11 | 2.21e+05 | 4.58e-05 | 1.71e-13 |
# `gtol` termination condition is satisfied.
# Number of iterations: 14, function evaluations: 10, CG iterations: 63, optimality: 4.58e-05, constraint violation: 1.71e-13, execution time: 1.4e+01 s.
# `gtol` termination condition is satisfied.

# analytic sparse |  14   |  10   |  63   | +6.9278e-11 | 2.21e+05 | 4.58e-05 | 1.71e-13 |
# Number of iterations: 14, function evaluations: 10, CG iterations: 63, optimality: 5.14e-05, constraint violation: 1.71e-13, execution time: 6e+01 s.

wpdiff =(A.dot(res.x) - wh) / wh * 100  # sum of state weights minus national weights
tpdiff = (targs(res.x, xmat, targets) - targets) / targets * 100  # pct diff
np.round(np.quantile(wpdiff, (0, .25, .5, .75, 1)), 2)
np.round(np.quantile(tpdiff, (0, .25, .5, .75, 1)), 2)
np.round(tpdiff, 2)
np.quantile(res.x, (0, .25, .5, .75, 1))


# %% test the hessian on the problem above
def hfun(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    # diffs = diffs * diff_weights
    grad = 2 * xmat.dot(diffs.T)
    return grad.flatten()


dw = np.ones(targets.shape)
dw_alt = dw * np.array([1, 2, 3, 4, 5, 6]).reshape((3, 2))
check = np.dot(dw_alt, dw_alt.T)
xbak = x0
2.76 * 2.76
x0 = np.ones(x0.size) * 5
rats = np.round(hfn(x0, xmat, targets, objscale, dw_alt) / hfn(x0, xmat, targets, objscale, dw), 2)
np.unique(rats)

f(x0, xmat, targets, objscale, dw)
gfun(x0, xmat, targets, objscale, dw)

# autograd hessian with diff weights that are 1
hvals = hfn(x0, xmat, targets, objscale, dw)
# hfn(x0*2, xmat, targets, objscale, dw) hessian is constant!
hvals.shape  # (s*h, s*h)
hvals
hvals[0:6, 0:4]
len(np.unique(hvals))  # 37 unique: s*h / 2 + zeroval
len(np.unique(np.round(hvals, 3)))  # s * h 22
# there appear to be 15 unique values in the 18 rows??

# hfn(x0, xmat, targets, objscale, dw3) / hfn(x0, xmat, targets, objscale, dw)
# times the square of diff_weights

# autograd hessian with diffweights that are not 1
hvals2 = hfn(x0, xmat, targets, objscale, diff_weights)
np.round(hvals2 / hvals, 8)
np.round(diff2, 8)
np.round(diff_weights, 8)

diff_weights * 1e3

hratios = hvals2 / hvals
np.unique(np.round(diff2, 8))
np.unique(np.round(hvals2 / hvals, 8))


np.round(hvals2, 3)[0:6, 0:6]
np.round(hesvals2, 3)

# my hessian
hesvals = 2 * np.dot(xmat, xmat.T)  # djb this is the key
hesvals # hessvals has 1 row per person
hesvals.shape
diff2 = np.square(diff_weights).flatten()
hesvals2 = hesvals * diff2
hesvals * diff2.T
hesvals2 = hesvals * diff2[:, None]
np.round(hesvals)
np.round(hesvals2, 3)
hesvals * np.square(diff_weights)[:, None]
A * b[:, None]
len(np.unique(hesvals))  # s*h / 2 unique values
# with 10,000 h and 50 s we appear to have 50,004,991 unique values
# I would have guessed s * h / 2 = 250k maybe plus the diagonal, or
# d = s * h
# (s * h - d) / 2 + d or (500k / 2) + maybe something?
#  500k variables so the hessian is 250 elements, 500k in a row
#

np.round(hesvals, 1)
# these vals fit into hessian as follows, one hesvec row at a time
# which diag, which cols -- note the repeats
#  0, 0:2x  1, 0:2x  2, 0:2x  3, 0:2x  4, 0:2x  5, 0:2
# -1, 3:5x  0, 3:5x  1, 3:5x  2, 3:5x  3, 3:5x  4, 3:5  # 1st item again
# -2, 6:8x  -1, 6:8x  0, 6:8x  1, 6:8x  2, 6:8x 3, 6:8
# -3, 9:11
# -4, 12:14
# -5, 15:17

# so now we know how to fill the full hessian, later worry
# about just a triangle, then worry about unique values
# each row of hesvec fills as follows:
# row 0: fills cols 0:2, starting rows 0:2, 3:5, 6:8, etc
# row 1:  cols 3:5, same sets of rows
# row5: cols 15:17, same sets of rows
# that will get us the full hessian

# slow fill
hmat = np.zeros((h * s, h * s))
# * np.square(diff_weights).flatten()
for valrow in range(0, hesvals.shape[0]):
    cols = range(valrow * s, valrow * s + s)
    print(valrow, cols)
    for valcol in range(0, hesvals.shape[1]):
        rows = range(valcol * s, valcol * s + s)
        print(rows)
        # hmat[rows, cols] = np.round(hesvals[valrow, valcol], 1)
        hmat[rows, cols] = hesvals[valrow, valcol]


hmat = np.zeros((h * s, h * s))
cols = range(3, 6)
rows = range(3, 6)
row = 2
hmat[rows, cols] = 1

hmat[0:7, 0:7]
np.round(hvals[0:7, 0:7], 1)

np.square(hmat - hvals).sum()



# %% ONETIME download Historical Table 2
files = [HT2_2018]

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% read and adjust Historical Table 2

ht2 = pd.read_csv(DOWNDIR + HT2_2018, thousands=',')
ht2
ht2.info()
ht2.STATE.describe()  # 54 -- states, DC, US, PR, OA
ht2.STATE.value_counts().sort_values()
ht2.groupby('STATE').STATE.count()  # alpha order
ht2.head()
ht2.columns.to_list()
# # convert all strings to numeric
# stn = ht2raw.columns.to_list()
# stn.remove('STATE')
# ht2[stn] = ht2raw[stn].apply(pd.to_numeric, errors='coerce', axis=1)
ht2

h2stubs = pd.DataFrame([
    [0, 'All income ranges'],
    [1, 'Under $1'],
    [2, '$1 under $10,000'],
    [3, '$10,000 under $25,000'],
    [4, '$25,000 under $50,000'],
    [5, '$50,000 under $75,000'],
    [6, '$75,000 under $100,000'],
    [7, '$100,000 under $200,000'],
    [8, '$200,000 under $500,000'],
    [9, '$500,000 under $1,000,000'],
    [10, '$1,000,000 or more']],
    columns=['h2stub', 'h2range'])
h2stubs
# h2stubs.info()

# create constants with state abbreviations
STATES_DCOAPRUS = ht2.groupby('STATE').STATE.count().index.tolist()
STATES = [x for x in STATES_DCOAPRUS if x not in ['DC', 'OA', 'PR', 'US']]
len(STATES)


# %% get reweighted national puf
PUF_RWTD = HDFDIR + 'puf2018_reweighted' + '.h5'
pufrw = pd.read_hdf(PUF_RWTD)  # 1 sec
pufrw.columns.sort_values()
pufrw


# %% prepare puf subset and weighted sums
# create a subset with ht2stub variable
pufsub = pufrw.copy()
pufsub['HT2_STUB'] = pd.cut(
    pufsub['c00100'],
    HT2_AGI_STUBS,
    labels=list(range(1, len(HT2_AGI_STUBS))),
    right=False)
pufsub.columns.sort_values().tolist()  # show all column names

pufsub[['pid', 'c00100', 'HT2_STUB', 'IRS_STUB']].sort_values(by='c00100')

# create list of target vars
alltargvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
alltargvars

pufsums = pufsub.groupby('HT2_STUB').apply(wsum,
                                            sumvars=alltargvars,
                                            wtvar='wtnew')
pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('HT2_STUB')
pufsums['HT2_STUB'] = pufsums.index
pufsums


# %% prepare compatible ht2 subset
# we'll rename HT2 columns to be like puf

# prepare a list of column names we want from ht2
ht2_all = list(HT2PUF_XWALK.keys())  # superset
ht2_all
prefixes = ('N0')  # must be tuple, not a list
ht2_use = [x for x in ht2_all if not x.startswith(prefixes)]
drops = ['N17000', 'A17000', 'A04470', 'A05800', 'A09600', 'A00700']
ht2_use = [x for x in ht2_use if x not in drops]
ht2_use
# ht2_use.remove('N17000')
ht2_use.append('STATE')
ht2_use.append('AGI_STUB')
ht2_use

# get those columns, rename as needed, and create new columns
ht2_sub = ht2.loc[:, ht2_use].copy()
ht2_sub = ht2_sub.rename(columns=HT2PUF_XWALK)
ht2_sub = ht2_sub.rename(columns={'AGI_STUB': 'HT2_STUB'})
# multiply dollar values by 1000
ht2_sub.columns
dollcols = ['c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e02400', 'c04800', 'irapentot']
dollcols
ht2_sub[dollcols] = ht2_sub[dollcols] * 1000
ht2_sub


# %% compare pufsums to HT2 for US
ht2sums = ht2_sub.query('STATE=="US"')
ht2sums = ht2sums.drop(columns=['STATE'])
ht2sums.columns

pufsums.columns

ht2sums
pufsums

pd.options.display.max_columns
pd.options.display.max_columns = 99
round(pufsums.drop(columns='HT2_STUB') / ht2sums.drop(columns='HT2_STUB') * 100 - 100)
pd.options.display.max_columns = 0
# e02400 is way off, c04800 has some trouble, and irapentot is way off, so don't use them
# the rest look good

# create adjustment ratios to apply to all ht2 values, based on national relationships
(pufsums.drop(columns='HT2_STUB') / ht2sums.drop(columns='HT2_STUB'))
(pufsums / ht2sums)

pufht2_ratios = (pufsums / ht2sums)
pufht2_ratios['HT2_STUB'] = pufht2_ratios.index
pufht2_ratios = pufht2_ratios.fillna(1)  # we won't use c04800
pufht2_ratios


# %% create adjusted HT2 targets for all states, based on the ratios
pufht2_ratios

# multiply each column of ht2_sub by its corresponding pufht2_ratios column
# is this the best way?
ht2_sub_adj = ht2_sub.copy()
ht2_sub_adjlong = pd.melt(ht2_sub_adj, id_vars=['HT2_STUB', 'STATE'])
ratios_long = pd.melt(pufht2_ratios, id_vars=['HT2_STUB'], value_name='ratio')
ht2_sub_adjlong =pd.merge(ht2_sub_adjlong, ratios_long, on=['HT2_STUB', 'variable'])
ht2_sub_adjlong['value'] = ht2_sub_adjlong['value'] * ht2_sub_adjlong['ratio']
ht2_sub_adjlong = ht2_sub_adjlong.drop(['ratio'], axis=1)
ht2_sub_adj = ht2_sub_adjlong.pivot(index=['HT2_STUB', 'STATE'], columns='variable', values='value')
# now we have an adjusted ht2 subset that has US totals equal to the puf totals

# check
ht2sumsadj = ht2_sub_adj.query('STATE=="US"')
pufsums / ht2sumsadj

# ht2_sub_adj.query('STATE=="OA"')

ht2_sub_adj = ht2_sub_adj.reset_index() # get indexes as columns


# %% choose a definition of targvars and targstates
targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00200',
            'e00300', 'e00600']

targvars = ['nret_all', 'nret_mars2', 'c00100', 'e00200',
            'e00300', 'e00600']

targvars = ['nret_all', 'nret_mars2', 'c00100', 'e00200', 'e00300']
targvars = ['nret_all', 'nret_mars2', 'c00100', 'e00200']
targvars = ['nret_all', 'nret_mars2', 'c00100']

targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100']
targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00600']

targvars = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00300', 'e00600']


targvars + ['HT2_STUB']

targstates = ['CA']
targstates = ['CA', 'FL', 'NY', 'TX']
targstates = ['CA']
targstates = ['CA', 'FL']
targstates = ['CA', 'NY']
targstates = ['CA', 'TX']
targstates = ['FL', 'TX']
targstates = ['FL', 'NY', 'TX']
targstates = ['CA', 'FL', 'NY', 'OH', 'PA', 'TX']
targstates = ['CA', 'CT', 'FL', 'GA', 'MA', 'NY', 'OH', 'OR', 'PA', 'TX', 'WA']

targstates = ['AL', 'AR', 'CA', 'CT', 'FL', 'GA', 'MA', 'MN', 'NJ', 'NY', 'OH', 'OR', 'PA', 'TX', 'WA']
targstates = ['AK', 'AL', 'AR', 'CA', 'CT', 'FL', 'GA', 'MD',
              'MA', 'MN', 'NH', 'NJ', 'NY', 'OH', 'OR', 'PA', 'TN', 'TX', 'VT', 'WA']

targstates = ['UT']

targstates = STATES

badstates = ['OA']
badstates = ['AK', 'OA']
badstates = ['AK', 'ND', 'SD', 'UT', 'WY', 'OA']
badstates = ['AK', 'ND', 'SD', 'WY', 'OA']
targstates = [x for x in STATES if x not in badstates]


# %% prepare a single stub for geoweighting
pufsub.columns
pufsub[['HT2_STUB', 'pid']].groupby(['HT2_STUB']).agg(['count'])

stub = 10
pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
pufstub

# use one of the following ht2_sub_adj
# ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub = ht2_sub_adj.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub
# show average target value per return times 100
round(ht2stub[targvars].div(ht2stub.nret_all, axis=0) * 100, 1)

# use one of the following
# htot = ht2sums.query('HT2_STUB ==@stub')[targvars]
htot = ht2sumsadj.query('HT2_STUB ==@stub')[targvars]

ptot = pufsums.query('HT2_STUB ==@stub')[targvars]
ptot / htot

# create an adjusted ht2stub that only has target states
ht2stub_adj = ht2stub.copy()
mask = np.logical_not(ht2stub_adj['STATE'].isin(targstates))
column_name = 'STATE'
ht2stub_adj.loc[mask, column_name] = 'XX'
ht2stub_adj[['STATE', 'HT2_STUB']].groupby(['STATE']).agg(['count'])
ht2stub_adj = ht2stub_adj.groupby(['STATE', 'HT2_STUB']).sum()
ht2stub_adj.info()
# average target value per return
round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
ht2stub_adj.sum()
ht2stub_adj
# pufsums.query('HT2_STUB == @stub')[targvars]
# np.array(ht2stub_adj.sum())
# ratios = pufsums.query('HT2_STUB == @stub')[targvars] / np.array(ht2stub_adj.sum())
# ratios = np.array(ratios)

# create possible starting values -- each record given each state's shares
ht2shares = ht2stub_adj.loc[:, ['nret_all']].copy()
ht2shares['share'] = ht2shares['nret_all'] / ht2shares['nret_all'].sum()
ht2shares = ht2shares.reset_index('STATE')

start_values = pufstub.loc[:, ['HT2_STUB', 'pid', 'wtnew']].copy().set_index('HT2_STUB')
# cartesian product
start_values = pd.merge(start_values, ht2shares, on='HT2_STUB')
start_values['iwhs'] = start_values['wtnew'] * start_values['share']
start_values  # good, everything is in the right order

iwhs = start_values.iwhs.to_numpy()  # initial weights, households and states

wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
xmat.shape
# use one of the following
# targets1 = ht2stub.drop(columns=['STATE', 'HT2_STUB'])
# targets = ht2stub_adj # .drop(columns=['STATE', 'HT2_STUB'])
targets = np.asarray(ht2stub_adj, dtype=float)
targets
targets.shape
# targets_scaled = targets * ratios
# targets.shape
# targets_scaled.shape

# targets_scaled / targets

# scale targets by ratio of pufsums to HT2


# %% poisson geo weighting
g = mw.Microweight(wh, xmat, targets)
# g = mw.Microweight(wh, xmat, targets_scaled)

# look at the inputs
g.wh
g.xmat
g.geotargets

# g.wh.shape
# g.xmat.shape
# g.geotargets.shape

# solve for state weights
g.geoweight()

# examine results
g.elapsed_minutes
np.round(g.result.fun.reshape(targets.shape), 1)
g.result  # this is the result returned by the solver
dir(g.result)
g.result.cost  # objective function value at optimum
g.result.message

# optimal values
g.beta_opt  # beta coefficients, s x k
g.delta_opt  # delta constants, 1 x h
g.whs_opt  # state weights
g.whs_opt.shape
g.geotargets_opt

g.geotargets_opt.sum(axis=0)

np.round(g.result.fun, 1)
np.round(g.result.fun.reshape(targets.shape), 1)
round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
# np.round(g.result.fun.reshape(7, 5), 1)


# %% constrained least squares geo weighting
# minimize the least squares diff, possibly using the result above as starting point
# with weight adding-up constraint

# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html#optimize-minimize-trustconstr

# DON'T USE fmin_slsqp - DOES NOT HANDLE CONSTRAINTS WELL
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html#scipy.optimize.fmin_slsqp
# https://www.programcreek.com/python/example/114542/scipy.optimize.fmin_slsqp

# show_options(solver="minimize", method="trust-ncg")

# constraints
# scipy.optimize.LinearConstraint(A, lb, ub, keep_feasible=False)[source]
# where:
# lb <= A.dot(x) <= ub
# A is (m, n) where m=number constraints, n=number variables
# this gives res[1] = A[1, 1] * x[1] + A[1, 2] * x[2] + ...A[1, n] * x[m]
# x is whs.flatten()
# lb, ub, are wh
# so we have h rows in A, whs.size columns in A
# A will have 1s in the appropriate places!!
# xx = p.whs.flatten() # xx[0:2] = p.whs [0, 0:2], xx[3:5] = p.whs[1, 0:2]
# A:
# 1 1 1 0000000000000000
# 0 0 0, 1 1 1, 000000000000
# 000, 000, 111, 0000000000

# lb <= A.dot(x) <= ub
# x is whs.flatten()
# lb, ub, are wh
# so we have h rows in A, whs.size columns in A

h = xmat.shape[0]
s = targets.shape[0]

pufsums.query('HT2_STUB==@stub')[targvars] / targets.sum(axis=0)

diff_weights = get_diff_weights(targets)
# diff_weights = np.ones(targets.shape)

# fast way to fill A
# get i and j indexes of nonzero values, and data
inz = np.arange(0, h).repeat(s)
jnz = np.arange(0, s * h)
A = sp.sparse.coo_matrix((np.ones(h*s), (inz, jnz)), shape=(h, h * s))
# A2.todense() - A.todense()

A = A.tocsr()  # csr format is faster for our calculations
# A = A.tocsc()
lincon = sp.optimize.LinearConstraint(A, wh, wh)
linconineq = sp.optimize.LinearConstraint(A, wh * .99, wh * 1.01)
linconineq_feas = sp.optimize.LinearConstraint(A, wh * .99, wh * 1.01, keep_feasible=True)

# objscale = 1e-6
wsmean = np.mean(wh) / targets.shape[0]
wsmin = np.min(wh) / targets.shape[0]
wsmax = np.max(wh)  # no state can get more than all of the national weight
xcheck = np.full(h * s, wsmean)
# xcheck = np.full(p.whs.size, 1)
# objscale = 1 / f(xcheck, xmat, targets, objscale=1, diff_weights=diff_weights) * 1e4
objscale = 1
objscale

# objective function at possible starting values
# f(xcheck, xmat, targets, objscale, diff_weights)
# f(iwhs, xmat, targets, objscale, diff_weights)

# bnds = sp.optimize.Bounds(0, np.inf)
bnds = sp.optimize.Bounds(0, wsmax)  # wsmin / 10

# p1 = np.ones(x0.size)
# hvp = hesspfn(x0, p1, xmat, targets, objscale, diff_weights)
# hvp
# hesspfn(res.x, p, xmat, targets, objscale, diff_weights)

# xmat.shape
# targets.shape
# diff_weights.shape
# bnds

# x0 = np.full(h * s, 1)
# x0 = g.whs_opt.flatten()
x0 = iwhs
# verify that starting values satisfy adding-up constraint
np.square(np.round(x0.reshape((h, s)).sum(axis=1) - wh, 2)).sum()
# end verification

# try alternative starting points
# x0 = np.ones(iwhs.size)
# x0 = np.zeros(iwhs.size)
# x0 = np.full(iwhs.size, iwhs.mean())
# x0 = np.full(iwhs.size, wsmax)

# equal shares - fast but little improvement, maybe try a lot of iterations
# x0 = np.ones((h, s)) / s
# x0 = np.multiply(x0, wh.reshape(x0.shape[0], 1)).flatten()

res2 = minimize(f, x0,
               method='trust-constr',
               bounds=bnds,
               constraints=linconineq,  # lincon linconineq linconineq_feas
               jac=gfun, # egfn, # gfun, # egfn
               # hess='2-point',  # 2-point 3-point cs
               hessp = f_hvp_wrap,
               args=(xmat, targets, 1, diff_weights),
               options={'maxiter': 50, 'verbose': 2,
                        'gtol': 1e-4, 'xtol': 1e-4,
                        'initial_tr_radius': 1,  # default 1
                        'factorization_method': 'AugmentedSystem'})  # default AugmentedSystem NormalEquation

# |  15   |  15   | 1244  | +2.6477e+02 | 9.17e+03 | 1.44e+00 | 1.51e+02 |

# |  15   |  15   |  39   | +1.8358e+04 | 8.09e-02 | 5.62e+00 | 9.23e-01 |
# |  20   |  20   |  44   | +1.8318e+04 | 2.78e-01 | 5.64e+00 | 9.23e-01 |

# |   8   |   8   |  416  | +3.6477e+01 | 2.76e+04 | 5.75e-02 | 3.17e+02 | state shares
# |  10   |  10   | 1116  | +4.1942e+01 | 2.76e+04 | 4.52e-02 | 3.14e+02 | state shares
# |  14   |  14   | 3055  | +4.0002e+01 | 2.76e+04 | 3.86e-02 | 2.10e+02 | state shares
# |   8   |   8   |  381  | +1.7041e+05 | 9.45e+04 | 6.73e+00 | 3.36e+03 | ones
# |  10   |  10   |  805  | +6.6573e+04 | 6.10e+05 | 2.64e+00 | 2.02e+04 | zeros
# |  100  |  100  |  99   | +2.3909e+05 | 1.56e+03 | 7.32e-01 | 9.09e-13 | equal shares, 250 secs
# |  100  |  100  |  99   | +2.3909e+05 | 1.56e+03 | 7.32e-01 | 9.09e-13 | equal shares 2-point, 250 secs
# |  10   |  10   |  471  | +2.1704e+05 | 1.01e+05 | 5.12e+00 | 5.16e+03 | iwhsmean
# |  10   |  10   |  439  | +1.5917e+10 | 2.66e+07 | 3.94e+02 | 1.05e+06 | wsmax


res = res2
wpdiff =(A.dot(res.x) - wh) / wh * 100  # sum of state weights minus national weights
tpdiff = (targs(res.x, xmat, targets) - targets) / targets * 100  # pct diff
np.round(np.quantile(wpdiff, (0, .1, .25, .5, .75, .9, 1)), 2)
np.round(np.quantile(tpdiff, (0, .1, .25, .5, .75, .9, 1)), 2)
np.round(tpdiff, 2)
np.round(np.quantile(res.x, (0, .1, .25, .5, .75, .9, 1)), 2)

res.execution_time / 60
res.fun
res.jac
res.message
res.nfev
res.nit
res.njev
res.status
res.x
res.x.min()
res.x.max()

# res = minimize(rosen, x0, method='SLSQP', jac=rosen_der,
#                constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True},
#                bounds=bounds)

# out ndarray of float The final minimizer of func.
# fx ndarray of float, if full_output is true The final value of the objective function.
# its int, if full_output is true The number of iterations.
# imode int, if full_output is true The exit mode from the optimizer (see below).
# smode string, if full_output is true Message describing the exit mode from the optimizer
# -1 : Gradient evaluation required (g & a)
#  0 : Optimization terminated successfully
#  1 : Function evaluation required (f & c)
#  2 : More equality constraints than independent variables
#  3 : More than 3*n iterations in LSQ subproblem
#  4 : Inequality constraints incompatible
#  5 : Singular matrix E in LSQ subproblem
#  6 : Singular matrix C in LSQ subproblem
#  7 : Rank-deficient equality constraint subproblem HFTI
#  8 : Positive directional derivative for linesearch
#  9 : Iteration limit reached


# %% ipopt
import src.reweight as rw

class cbacks(object):
    """
    Must have:
        objective
        constraints
        gradient
        jacobian
        jacobianstructure
        hessian
        hessianstructure
        intermediate
    """
    def __init__(self, cc, quiet):
        self._cc = cc
        self._n = cc.shape[0]
        self._m = cc.shape[1]
        self._quiet = quiet

    def objective(self, x):
        """Calculate objective function."""
        return np.sum((x - 1)**2)

    def constraints(self, x):
        return np.dot(x, self._cc)

    def gradient(self, x):
        """Calculate gradient of objective function."""
        return 2 * x - 2


    def jacobian(self, x):
        row, col = self.jacobianstructure()
        return self._cc.T[row, col]

    def jacobianstructure(self):
        return np.nonzero(self._cc.T)

    def hessian(self, x, lagrange, obj_factor):
        H = np.full(self._n, 2) * obj_factor
        return H

    def hessianstructure(self):
        """
        Row and column indexes of nonzero elements of hessian.

        A tuple of two arrays: one for row indexes and one for column indexes.
        In this problem the hessian has nonzero elements only on the diagonal
        so this returns an array of row indexes of arange(0, n) where n is
        the number of rows (and columns) in the square hessian matrix, and
        the same array for the column index column indexes.

        These indexes must correspond to the order of the elements returned
        from the hessian function. That requirement is enforced in that
        function.

        Note: The cyipopt default hessian structure is a lower triangular
        matrix, so if that is what the hessian function produces, this
        function is not needed.

        Returns
        -------
        hstruct : tuple:
            First array has row indexes of nonzero elements of the hessian
            matrix.
            Second array has column indexes for these elements.

        """
        hidx = np.arange(0, self._n, dtype='int64')
        hstruct = (hidx, hidx)
        return hstruct

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        """
        Print intermediate results after each iteration.

        Parameters
        ----------
        alg_mod : TYPE
            DESCRIPTION.
        iter_count : TYPE
            DESCRIPTION.
        obj_value : TYPE
            DESCRIPTION.
        inf_pr : TYPE
            DESCRIPTION.
        inf_du : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.
        d_norm : TYPE
            DESCRIPTION.
        regularization_size : TYPE
            DESCRIPTION.
        alpha_du : TYPE
            DESCRIPTION.
        alpha_pr : TYPE
            DESCRIPTION.
        ls_trials : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # print("Objective value at iteration #%d is - %g"
        #     % (iter_count, obj_value))
        if(not self._quiet):
            print("Iter, obj, infeas #%d %g %g"
                  % (iter_count, obj_value, inf_pr))

def get_ccscale(cc, ccgoal, method='mean'):
# use mean or median as the denominator
    if(method == 'mean'):
        denom = cc.sum(axis=0) / cc.shape[0]
        elif(method == 'median'):
            denom = np.median(cc, axis=0)
        ccscale = np.absolute(ccgoal / denom)
        # ccscale = ccscale / ccscale
        return ccscale

cc = (xmat.T * wh).T

# scale constraint coefficients and targets
ccgoal = 1
ccscale = get_ccscale(cc, ccgoal=ccgoal, method='mean')
# print(ccscale)
# ccscale = 1
cc = cc * ccscale  # mult by scale to have avg derivative meet our goal
targets = targets * ccscale

# IMPORTANT: define callbacks AFTER we have scaled cc and targets
# because callbacks must be initialized with scaled cc
callbacks = rw.Reweight_callbacks(cc, quiet)

# x vector starting values, and lower and upper bounds
x0 = np.ones(self._n)
lb = np.full(self._n, xlb)
ub = np.full(self._n, xub)

# constraint lower and upper bounds
cl = targets - abs(targets) * crange
cu = targets + abs(targets) * crange

nlp = ipopt.problem(
            n=n,
            m=m,
            problem_obj=callbacks,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu)

# objective function scaling
# objscale = self.get_objscale(objgoal=objgoal, xbase=1.2)
# print(objscale)
nlp.addOption('obj_scaling_factor', 1)  # multiplier

# define additional options as a dict
opts = {'print_level': 5,
        'file_print_level': 5,
        # 'jac_d_constant': 'yes',
        'hessian_constant': 'yes',
        'max_iter': 10,
        'mumps_mem_percent': 100,  # default 1000
        'linear_solver': 'mumps',
        }
x, info = nlp.solve(x0)

# # TODO: check against already set options, etc. see ipopt_wrapper.py
for option, value in opts.items():
    nlp.addOption(option, value)



# nlp.addOption('nlp_scaling_method', 'equilibration-based')
        # outfile = 'test4.out'
        # if os.path.exists(outfile):
        #     os.remove(outfile)
        # nlp.addOption('output_file', outfile)
        # nlp.addOption('derivative_test', 'first-order')  # second-order

        # nlp_scaling_method: default gradient-based
        # equilibration-based needs MC19
        # nlp.addOption('nlp_scaling_method', 'equilibration-based')
        # nlp.addOption('nlp_scaling_max_gradient', 1e-4)  # 100 default
        # nlp.addOption('mu_strategy', 'adaptive')  # not good
        # nlp.addOption('mehrotra_algorithm', 'yes')  # not good
        # nlp.addOption('mumps_mem_percent', 100)  # default 1000
        # nlp.addOption('mumps_pivtol', 1e-4)  # default 1e-6; 1e-2 is SLOW
        # nlp.addOption('mumps_scaling', 8)  # 77 default




# %% play -- hessian

# rough -- define hessian, etc
# it is constant so get its value at a point and
ediag_vals = ehfn(iwhs, xmat, targets, objscale, diff_weights)

H = nd.Hessian(f)
hvals = H(x0, xmat, targets, objscale, diff_weights)
hvals2 = hfn(x0, xmat, targets, objscale, diff_weights)

hvals.shape # (h x s, h xs) 60, 60
hvals[0:8, 0:3]
# hest = 2 * np.square(xmat).sum(axis=1)
hest = 2 * np.dot(xmat, xmat.T)
np.round(hvals[0:8,0:5], 2)
np.round(hvals2[0:8,0:5], 2)
hest
np.round(hvals.diagonal(), 1)
np.round(hest.diagonal(), 1)
np.multiply(xmat, xmat)

hest.shape
hvals.shape
np.round(hest[0:5, 0:5], 1)
np.round(hvals[0:12, 0:4], 1)

ma = np.identity(3)

2 * np.square(xmat[0:2, 0:2])

2 * np.square(xmat).sum(axis=1)

z1 = egfn(x0, xmat, targets, objscale, diff_weights)
z2 = gfun(x0, xmat, targets, objscale, diff_weights)
z1.shape
z2.shape
z1.sum()
z2.sum()
np.square(z2 - z1).sum()



# %% cvxpy
# https://www.cvxpy.org/



def diffs(x, xmat, targets):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    return(diffs)

p = mtp.Problem(h=20, s=3, k=2)
x = cp.Variable(p.whs.size)
# constraints = [0 <= x, x <= 1]
# objective = cp.Minimize(cp.sum_squares(A*x - b))


# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A@x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)


# %% get new national weights - loop through all states in a stub and calculate weights
# prepare the stub
stub = 9
pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
# begin the loop (must loopify)
# create an adjusted ht2stub that only has target state
tstate = 'NY'
ht2stub_adj = ht2stub.copy()
mask = np.logical_not(ht2stub_adj['STATE'] == tstate)
column_name = 'STATE'
ht2stub_adj.loc[mask, column_name] = 'XX'
# ht2stub_adj[['STATE', 'HT2_STUB']].groupby(['STATE']).agg(['count'])
ht2stub_adj = ht2stub_adj.groupby(['STATE', 'HT2_STUB']).sum()
# ht2stub_adj.info()
# ht2stub_adj
# average target value per return
# round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
# ht2stub_adj.sum()
# ht2stub_adj
# pufsums.query('HT2_STUB == @stub')[targvars]
# np.array(ht2stub_adj.sum())
# ratios = pufsums.query('HT2_STUB == @stub')[targvars] / np.array(ht2stub_adj.sum())
# ratios = np.array(ratios)
# looks like int income in OA is a problem

wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
# xmat.shape

targets = np.asarray(ht2stub_adj, dtype=float)
targets

g = mw.Microweight(wh, xmat, targets)
# g = mw.Microweight(wh, xmat, targets_scaled)

# look at the inputs
g.wh
g.xmat
g.geotargets

# solve for state weights
g.geoweight()
# g.beta_opt
# g.delta_opt  # delta constants, 1 x h
pd.DataFrame(g.beta_opt.flatten()).describe()
pd.DataFrame(g.delta_opt).describe()

# After 10 iterations
# C:\programs_python\weighting\src\microweight.py:103: RuntimeWarning: overflow encountered in exp
#   beta_x = np.exp(np.dot(beta, xmat.T))
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: divide by zero encountered in true_divide
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: overflow encountered in true_divide
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\programs_python\weighting\src\microweight.py:105: RuntimeWarning: divide by zero encountered in log
#   delta = np.log(wh / beta_x.sum(axis=0))  # axis=0 gives colsums
# C:\Users\donbo\anaconda3\envs\analysis\lib\site-packages\numpy\core\_methods.py:47: RuntimeWarning: overflow encountered in reduce
#   return umr_sum(a, axis, dtype, out, keepdims, initial, where)

# examine results
g.elapsed_minutes

# it must be possible to make percent errors better
np.round(g.result.fun.reshape(targets.shape), 1)
# x0 values and results
# x0 0
# array([[  5.2, -13.6,   8.8,   1.4,  -5.2,   0.7,   0.9],
#        [  0.2,   5.2,  -0.5,   0.6,   1.9,   5.6,   1.6]])
# x0 1e-9
# array([[ 2.1,  0.3, -0.5, -1.3,  0.1,  0.6,  0.2],
#        [ 0.5,  3.4,  0.3,  0.8,  1.3,  5.6,  1.7]])
# x0 1e-12
# array([[ 0.7,  0.3, -0.3, -0.1,  0.1,  0.6,  0.2],
#        [ 0.6,  3.4,  0.3,  0.7,  1.3,  5.6,  1.7]])


g.result  # this is the result returned by the solver
dir(g.result)
g.result.cost  # objective function value at optimum
g.result.message


# %% geoweight testing and practice
mtp.Problem.help()

p = mtp.Problem(h=10000, s=50, k=10)
# p = mtp.Problem(h=20000, s=30, k=10)  # moderate-sized problem, < 1 min

# I don't think our problems for a single AGI range will get bigger
# than the one below:
#   30k tax records, 50 states, 30 characteristics (targets) per state
# but problems will be harder to solve with real data
# p = mtp.Problem(h=30000, s=50, k=30) # took 31 mins on my computer

mw.Microweight.help()

p.xmat.shape
p.targets.shape
g1 = mw.Microweight(p.wh, p.xmat, p.targets)

# look at the inputs
g1.wh
g1.xmat
g1.geotargets

# solve for state weights
g1.geoweight()

# examine results
g1.elapsed_minutes
g1.result  # this is the result returned by the solver
dir(g1.result)
g1.result.cost  # objective function value at optimum
g1.result.message

# optimal values
g1.beta_opt  # beta coefficients, s x k
g1.delta_opt  # delta constants, 1 x h
g1.whs_opt  # state weights
g1.geotargets_opt




# %% Peter's  crosswalks
# Peter's mappings of puf to historical table 2
# "n1": "N1",  # Total population
# "mars1_n": "MARS1",  # Single returns number
# "mars2_n": "MARS2",  # Joint returns number
# "c00100": "A00100",  # AGI amount
# "e00200": "A00200",  # Salary and wage amount
# "e00200_n": "N00200",  # Salary and wage number
# "c01000": "A01000",  # Capital gains amount
# "c01000_n": "N01000",  # Capital gains number
# "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
# "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
# "c17000": "A17000",  # Medical expenses deducted amount
# "c17000_n": "N17000",  # Medical expenses deducted number
# "c04800": "A04800",  # Taxable income amount
# "c04800_n": "N04800",  # Taxable income number
# "c05800": "A05800",  # Regular tax before credits amount
# "c05800_n": "N05800",  # Regular tax before credits amount
# "c09600": "A09600",  # AMT amount
# "c09600_n": "N09600",  # AMT number
# "e00700": "A00700",  # SALT amount
# "e00700_n": "N00700",  # SALT number

    # Maps PUF variable names to HT2 variable names
VAR_CROSSWALK = {
    "n1": "N1",  # Total population
    "mars1_n": "MARS1",  # Single returns number
    "mars2_n": "MARS2",  # Joint returns number
    "c00100": "A00100",  # AGI amount
    "e00200": "A00200",  # Salary and wage amount
    "e00200_n": "N00200",  # Salary and wage number
    "c01000": "A01000",  # Capital gains amount
    "c01000_n": "N01000",  # Capital gains number
    "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
    "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
    "c17000": "A17000",  # Medical expenses deducted amount
    "c17000_n": "N17000",  # Medical expenses deducted number
    "c04800": "A04800",  # Taxable income amount
    "c04800_n": "N04800",  # Taxable income number
    "c05800": "A05800",  # Regular tax before credits amount
    "c05800_n": "N05800",  # Regular tax before credits amount
    "c09600": "A09600",  # AMT amount
    "c09600_n": "N09600",  # AMT number
    "e00700": "A00700",  # SALT amount
    "e00700_n": "N00700",  # SALT number
}


# %% OLD items


def gfun_old(x, xmat, targets, objscale, diff_weights):
    whs = x.reshape((xmat.shape[0], targets.shape[0]))
    diffs = np.dot(whs.T, xmat) - targets
    # diffs = diffs * diff_weights
    grad = 2 * xmat.dot(diffs.T)
    return grad.flatten()


# %% OLD constraint notes
# A = lil_matrix((h, h * s))
# look for a faster way to do this next for loop
# for i in range(0, h):
#     A[i, range(i*s, i*s + s)] = 1
# A
# b=A.todense()  # ok to look at dense version if small
# A = A.tocsr()  # csr format is faster for our calculations
# lincon = sp.optimize.LinearConstraint(A, wh, wh)
# keep_feasible doesn't seem to matter
# lincon_feas = sp.optimize.LinearConstraint(A, wh, wh, keep_feasible=True)
# lincon = sp.optimize.LinearConstraint(A, wh*.98, wh*1.02)  # range around weight sums