# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:38:57 2020

@author: donbo
"""

# %% imports
import pandas as pd
import sys
import puf_constants as pc


# %% weighted percentiles
# def weighted_percentile(data, weights, perc):
#     """
#     perc : percentile in [0-1]!
#     """
#     ix = np.argsort(data)
#     data = data[ix] # sort data
#     weights = weights[ix] # sort weights
#     cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
#     return np.interp(perc, cdf, data)


# def quantile_at_values(values, population, weights=None):
#     values = np.atleast_1d(values).astype(float)
#     population = np.atleast_1d(population).astype(float)
#     # if no weights are given, use equal weights
#     if weights is None:
#         weights = np.ones(population.shape).astype(float)
#         normal = float(len(weights))
#     # else, check weights
#     else:
#         weights = np.atleast_1d(weights).astype(float)
#         assert len(weights) == len(population)
#         assert (weights >= 0).all()
#         normal = np.sum(weights)
#         assert normal > 0.
#     quantiles = np.array([np.sum(weights[population <= value]) for value in values]) / normal
#     assert (quantiles >= 0).all() and (quantiles <= 1).all()
#     return quantiles

# define a function for weighted quantiles. input: x, q
# x: two-column data, the second column is weight. q: percentile
# def wquantile(xvalues, weights, qtile):
#     x = pd.DataFrame(data={'data': xvalues, 'weights': weights})
#     xsort = x.sort_values(x.columns[0])
#     xsort['index'] = range(len(x))
#     p = qtile * x[x.columns[1]].sum()
#     pop = float(xsort[xsort.columns[1]][xsort['index']==0])
#     i = 0
#     while pop < p:
#         pop = pop + float(xsort[xsort.columns[1]][xsort['index']==i+1])
#         i = i + 1
#     return xsort[xsort.columns[0]][xsort['index']==i]


def wp(data, wt, percentiles):
    """Compute weighted percentiles.
    If the weights are equal, this is the same as normal percentiles.
    Elements of the C{data} and C{wt} arrays correspond to
    each other and must have equal length (unless C{wt} is C{None}).
    http://kochanski.org/gpk/code/speechresearch/gmisclib/gmisclib.weighted_percentile-pysrc.html#wp

    @param data: The data.
    @type data: A L{np.ndarray} array or a C{list} of numbers.
    @param wt: How important is a given piece of data.
    @type wt: C{None} or a L{np.ndarray} array or a C{list} of numbers.
            All the weights must be non-negative and the sum must be
            greater than zero.
    @param percentiles: what percentiles to use.  (Not really percentiles,
            as the range is 0-1 rather than 0-100.)
    @type percentiles: a C{list} of numbers between 0 and 1.
    @rtype: [ C{float}, ... ]
    @return: the weighted percentiles of the data.
    """
    assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero"
    assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one"
    data = np.asarray(data)
    assert len(data.shape) == 1
    if wt is None:
        wt = np.ones(data.shape, np.float)
    else:
        wt = np.asarray(wt, np.float)
        assert wt.shape == data.shape
        assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."

    assert len(wt.shape) == 1
    n = data.shape[0]
    assert n > 0
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    aw = np.add.accumulate(sw)
    if not aw[-1] > 0:
        print("Nonpositive weight sum")
    w = (aw-0.5*sw)/aw[-1]
    spots = np.searchsorted(w, percentiles)
    o = []
    for (s, p) in zip(spots, percentiles):
        if s == 0:
            o.append(sd[0])
        elif s == n:
            o.append(sd[n-1])
        else:
            f1 = (w[s] - p)/(w[s] - w[s-1])
            f2 = (p - w[s-1])/(w[s] - w[s-1])
            assert f1 >= 0 and f2 >= 0 and f1 <= 1 and f2 <= 1
            assert abs(f1+f2-1.0) < 1e-6
            o.append(sd[s-1]*f1 + sd[s]*f2)
    return o



# %% function to create mask identifying filers

def filers(puf, year=2017):
    """Return boolean array identifying tax filers.

    Parameters
    ----------
    puf : TYPE
        DESCRIPTION.
    year : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    # IRS rules for filers: https://www.irs.gov/pub/irs-prior/p17--2017.pdf

    Gross income. This includes all income you receive in the form of money,
    goods, property, and services that isn't exempt from tax. It also includes
    income from sources outside the United States or from the sale of your main
    home (even if you can exclude all or part of it). Include part of your
    social security benefits if: 1. You were married, filing a separate return,
    and you lived with your spouse at any time during 2017; or 2. Half of your
    social security benefits plus your other gross income and any tax-exempt
    interest is more than $25,000 ($32,000 if married filing jointly).

    define gross income as above the line income plus any losses deducted in
    arriving at that, plus any income excluded in arriving at that
    """
    if year == 2017:
        s_inc_lt65 = 10400
        s_inc_ge65 = 11950

        mfj_inc_bothlt65 = 20800
        mfj_inc_onege65 = 22050
        mfj_inc_bothge65 = 23300

        mfs_inc = 4050

        hoh_inc_lt65 = 13400
        hoh_inc_ge65 = 14950

        qw_inc_lt65 = 16750
        qw_inc_ge65 = 18000

        wage_threshold = 1000

    # above the line income is agi plus above line adjustments getting to agi
    above_line_income = puf.c00100 + puf.c02900

    # gross_income FOR NOW assume same as above-the-line income
    gross_income = above_line_income

    # to be on the safe side, don't let gross_income be negative
    gross_income = gross_income * gross_income.ge(0)

    # define filer masks
    # households that are required to file based on marital status,
    # age, and gross income

    # single
    m_single_lt65 = puf.MARS.eq(1) \
        & puf.age_head.lt(65) \
        & gross_income.ge(s_inc_lt65)

    m_single_ge65 = puf.MARS.eq(1) \
        & puf.age_head.ge(65) \
        & gross_income.ge(s_inc_ge65)

    m_single = m_single_lt65 | m_single_ge65

    # married joint
    m_mfj_bothlt65 = puf.MARS.eq(2) \
        & puf.age_head.lt(65) \
        & puf.age_spouse.lt(65) \
        & gross_income.ge(mfj_inc_bothlt65)

    m_mfj_onege65 = puf.MARS.eq(2) \
        & (puf.age_head.ge(65) | puf.age_spouse.ge(65)) \
        & ~(puf.age_head.ge(65) & puf.age_spouse.ge(65)) \
        & gross_income.ge(mfj_inc_onege65)

    m_mfj_bothge65 = puf.MARS.eq(2) \
        & puf.age_head.ge(65) \
        & puf.age_spouse.ge(65) \
        & gross_income.ge(mfj_inc_bothge65)

    m_mfj = m_mfj_bothlt65 | m_mfj_onege65 | m_mfj_bothge65

    # married separate
    m_mfs = puf.MARS.eq(3) & gross_income.ge(mfs_inc)

    # head of household
    m_hoh_lt65 = puf.MARS.eq(4) \
        & puf.age_head.lt(65) \
        & gross_income.ge(hoh_inc_lt65)

    m_hoh_ge65 = puf.MARS.eq(4) \
        & puf.age_head.ge(65) \
        & gross_income.ge(hoh_inc_ge65)

    m_hoh = m_hoh_lt65 | m_hoh_ge65

    # qualifying widow(er)
    m_qw_lt65 = puf.MARS.eq(5) \
        & puf.age_head.lt(65) \
        & gross_income.ge(qw_inc_lt65)

    m_qw_ge65 = puf.MARS.eq(5) \
        & puf.age_head.ge(65) \
        & gross_income.ge(qw_inc_ge65)

    m_qw = m_qw_lt65 | m_qw_ge65

    m_required = m_single | m_mfj | m_mfs | m_hoh | m_qw

    # returns that surely will or must file even if
    # marital-status/age/gross_income requirement is not met
    m_negagi = puf.c00100.lt(0)  # negative agi
    m_iitax = puf.iitax.ne(0)
    m_credits = puf.c07100.ne(0) | puf.refund.ne(0)
    m_wages = puf.e00200.ge(wage_threshold)

    m_likely = m_negagi | m_iitax | m_credits | m_wages

    m_filer = m_required | m_likely

    return m_filer


# %% prepare puf for comparison
def prep_puf(puf, pufvars_to_nnz=None):
    puf['common_stub'] = pd.cut(
        puf['c00100'],
        pc.COMMON_STUBS,
        labels=range(1, 19),
        right=False)

    puf['ht2_stub'] = pd.cut(
        puf['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)

    puf['filer'] = filers(puf)

    puf['nret_all'] = 1

    # marital status indicators
    puf['mars1'] = puf.MARS.eq(1)
    puf['mars2'] = puf.MARS.eq(2)
    puf['mars3'] = puf.MARS.eq(3)
    puf['mars4'] = puf.MARS.eq(4)
    puf['mars5'] = puf.MARS.eq(5)

    # create capital gains positive and negative
    puf['c01000pos'] = puf.c01000 * puf.c01000.gt(0)
    puf['c01000neg'] = puf.c01000 * puf.c01000.lt(0)

    # create partnership and S corp e26270 positive and negative
    puf['e26270pos'] = puf.e26270 * puf.e26270.gt(0)
    puf['e26270neg'] = puf.e26270 * puf.e26270.lt(0)

    if pufvars_to_nnz is not None:
        for var in pufvars_to_nnz:
            puf[var + '_nnz'] = puf[var].ne(0) * 1

    return puf


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


def xlrange(io, sheet_name=0,
            firstrow=1, lastrow=None,
            usecols=None, colnames=None):
    # firstrow and lastrow are 1-based
    if colnames is None:
        if usecols is None:
            colnames = None
        elif isinstance(usecols, list):
            colnames = usecols
        else:
            colnames = usecols.split(',')
    nrows = None
    if lastrow is not None:
        nrows = lastrow - firstrow + 1
    df = pd.read_excel(io,
                       header=None,
                       names=colnames,
                       usecols=usecols,
                       skiprows=firstrow - 1,
                       nrows=nrows)
    return df

