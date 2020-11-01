# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 04:19:27 2020

@author: donbo
"""

# %% target notes
# puf counts generally are my definition of filers
# in addition to the obvious items

# filing status

# income items

# agi and c00100, good
# wages and e00200, good

# taxable interest
# taxint irs line 8a, Form 1040, same as e00300
# DON'T TARGET NUMBER OF NZ RETURNS

# ordinary dividends line 9a form 1040
# orddiv same as e00600 the larger one
# looks like I should reduce the growfactor then should be ok to target nz and sum

# capital gains
# cgnet and c01000
# should be good to target both

# pensions total
# pensions e01500
# amount good, too many nz in puf but prob should target both

# social security
# socsectaxable  c02500
# looks like I should target both nz and amount
# target taxable, not total

# itemized deductions
# irs id_medical_capped c17000 the limited deduction in the irs data

# taxes paid
# id_taxpaid
# irs 17in21id.xls taxes paid deduction  46,431,232 624,820,806
# c18300 Sch A: State and local taxes plus real estate taxes deducted (component of pre-limitation c21060 total)
# c18300  45,959,926 585,237,496,064
# wouldn't hurt to increase the growfactor




# %% imports
import pandas as pd
import numpy as np
import puf_constants as pc


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
HDFDIR = r'C:\programs_python\puf_analysis\ignore/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'

BASE_NAME = 'puf_adjusted'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast


# %% constants
# pc.HT2_AGI_STUBS
# pc.ht2stubs
# pc.IRS_AGI_STUBS
# pc.irsstubs

qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)

# %% misc functions
def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    """
    ix = np.argsort(data)
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    return np.interp(perc, cdf, data)


def quantile_at_values(values, population, weights=None):
    values = np.atleast_1d(values).astype(float)
    population = np.atleast_1d(population).astype(float)
    # if no weights are given, use equal weights
    if weights is None:
        weights = np.ones(population.shape).astype(float)
        normal = float(len(weights))
    # else, check weights
    else:
        weights = np.atleast_1d(weights).astype(float)
        assert len(weights) == len(population)
        assert (weights >= 0).all()
        normal = np.sum(weights)
        assert normal > 0.
    quantiles = np.array([np.sum(weights[population <= value]) for value in values]) / normal
    assert (quantiles >= 0).all() and (quantiles <= 1).all()
    return quantiles

# define a function for weighted quantiles. input: x, q
# x: two-column data, the second column is weight. q: percentile
def wquantile(xvalues, weights, qtile):
    x = pd.DataFrame(data={'data': xvalues, 'weights': weights})
    xsort = x.sort_values(x.columns[0])
    xsort['index'] = range(len(x))
    p = qtile * x[x.columns[1]].sum()
    pop = float(xsort[xsort.columns[1]][xsort['index']==0])
    i = 0
    while pop < p:
        pop = pop + float(xsort[xsort.columns[1]][xsort['index']==i+1])
        i = i + 1
    return xsort[xsort.columns[0]][xsort['index']==i]


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

    # add back any losses that were used to reduce above the line income
    # these are negative so we will subtract them from above the line income
    capital_losses = puf.c23650.lt(0) * puf.c23650 \
        + puf.c01000.lt(0) * puf.c01000
    other_losses = puf.e01200.lt(0) * puf.e01200
    business_losses = puf.e00900.lt(0) * puf.e00900
    rent_losses = puf.e02000.lt(0) * puf.e02000
    farm_losses = puf.e02100.lt(0) * puf.e02100
    above_line_losses = capital_losses + other_losses + business_losses \
        + rent_losses + farm_losses

    # add back any untaxed income that was excluded in calculating
    # above the line income
    interest_untaxed = puf.e00400
    dividends_untaxed = puf.e00600 - puf.e00650
    pensions_untaxed = puf.e01500 - puf.e01700  # always ge zero, I checked
    # socsec_untaxed is OVERSTATED - I think IRS has a limit on amount
    socsec_untaxed = puf.e02400 - puf.c02500  # always ge zero, I checked
    above_line_untaxed = interest_untaxed + dividends_untaxed \
        + pensions_untaxed + socsec_untaxed

    gross_income = above_line_income - above_line_losses + above_line_untaxed

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


# %% get target data
IRSDAT = DATADIR + 'targets2017_collapsed.csv'
irstot = pd.read_csv(IRSDAT)
irstot
irstot.info()
irstot.count()
irsvars = irstot.variable.value_counts().sort_index()


# %% create cgnet and nret_cgnet
# cgnet = cggross - cgloss, we will match against c01000
# we will calculate nret_cgnet ASSUMING it is nret_cggross + nret_cgloss
cgvars = ['cggross', 'cgloss', 'nret_cggross', 'nret_cgloss']
capgains = irstot.query('variable in @cgvars')
idvars = ['src', 'common_stub', 'incrange', 'table_description']
keepvars = idvars + ['variable', 'value']
cgwide = capgains.loc[:, keepvars].pivot(index=idvars, columns=['variable'], values='value')
cgwide['cgnet'] = cgwide.cggross - cgwide.cgloss
cgwide['nret_cgnet'] = cgwide.nret_cggross + cgwide.nret_cgloss
# cgwide = cgwide.rename(columns={'nret_cggross': 'nret_cgnet'}).reset_index()
cgwide = cgwide.reset_index()
cgwide = cgwide.drop(columns=['cggross', 'cgloss'])
cglong = cgwide.melt(id_vars=idvars)

# set column_description
# faster approach
ret_lab = 'Number of returns with capital gains net taxable ASSUMED = nret_cggross + nret_cgloss'
val_lab = 'Capital gains net taxable CALCULATED as gross - loss'
cglong.loc[cglong['variable'] == 'nret_cgnet', 'column_description'] = ret_lab
cglong.loc[cglong['variable'] == 'cgnet', 'column_description'] =val_lab

# alternative easy to understand but slow approach
# def f(row):
#     # not vectorized, only good for small data frames
#     if row['variable'] == 'cgnet':
#         label = 'Capital gains net taxable CALCULATED as gross - loss'
#     elif row['variable'] == 'nret_cgnet':
#         label = 'Number of returns with capital gains net taxable ASSUMED = nret_cggross'
#     return label
# cglong['column_description'] = cglong.apply(f, axis=1)


# %% update irstot
irstot = irstot.append(cglong)


# %% get the puf
puf = pd.read_hdf(IGNOREDIR + 'puf2017_2020-10-26.h5')  # 1 sec
puf['common_stub'] = pd.cut(
    puf['c00100'],
    pc.COMMON_STUBS,
    labels=range(1, 19),
    right=False)
puf.info()


# %% define filers
# age_head, age_spouse
# np.quantile(puf.age_head, qtiles)  # 1 to 85 (1??)
# np.quantile(puf.age_spouse, qtiles)  # 50 to 97
# puf.MARS.value_counts()

puf['filer'] = filers(puf, 2017)
puf.filer.sum()
pufvars = puf.columns.sort_values()


# %% scratch area helper functions
def wsum(var, puf):
    val = (puf[var] * puf['s006']).sum()
    return val

def nret(var, puf):
    val = ((puf[var] != 0)* puf['s006']).sum()
    return val

def irsn(irsvar):
    irsvar = 'nret_' + irsvar
    q = 'common_stub==0 and variable==@irsvar'
    val = irstot.query(q)[['value']]
    return val.iat[0,0]

def irssum(irsvar):
    q = 'common_stub==0 and variable==@irsvar'
    val = irstot.query(q)[['value']]
    return val.iat[0,0]

# %% scratch taxable interest
# per pub 1304
# irs concept (line 8a, Form 1040), same as e00300
# we have way too many records with nonzero values but the weighted sum is
# in the right ballpark (4.6%)
# DON'T TARGET NUMBER OF NZ RETURNS

data = puf[puf.filer].e00300
weights = puf[puf.filer].s006

data = puf.query('filer and (e00300 != 0)')['e00300']
weights = puf.query('filer and (e00300 != 0)')['s006']

res = wp(data, weights, qtiles)
np.round(res, 1)
qtiles
type(res)



# %% scratch dividends
# irs ordinary dividends line 9a form 1040 same as e00600 this is the larger one so target this
# irs Qualified Dividends (line 9b, Form 1040) same as e00650
var = 'e00650'
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')


# %% scratch
# e02000 Sch E total rental, royalty, partnership, S-corporation, etc, income/loss (includes e26270 and e27200)
# e26270 Sch E: Combined partnership and S-corporation net income/loss (includes k1bx14p and k1bx14s amounts and is included in e02000)

# nret_partnerscorpinc Number of returns with partnership or S corporation net income
# partnerscorpinc Partnership or S corporation net income
# nret_partnerscorploss Number of returns with partnership or S corporation net loss
# partnerscorploss Partnership or S corporation net loss



# %% scratch medical deductions
# id_medical_capped
# e17500 Description: Itemizable medical and dental expenses. WARNING: this variable is zero below the floor in PUF data.
# c17000 Sch A: Medical expenses deducted (component of pre-limitation c21060 total)
# I don't understand why c17000 is called component of pre-limitation -- it does appear to be limited
# e17500_capped Sch A: Medical expenses, capped as a decimal fraction of AGI
# irs 17in21id.xls pre-limit total 155,408,904  10,171,257
# id_medical_capped irs 17in21id.xls limited 102,533,387  10,171,257

irsvar = 'id_medical_capped'
print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

var = 'e17500' # 200,511,523,398  17,563,931
var = 'c17000'  # 96,675,292,760  9,725,100
var = 'e17500_capped'  # 200,511,523,398 17,563,931
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')

# seems like I should match c17000 against id_medical_capped the limited deduction in the irs data


# %% scratch SALT
var = 'c18300'  # 45,959,926 585,237,496,064
var = 'e18400'  # 153,487,189 513,295,087,752
var = 'e18400_capped' # 153,487,189 513,295,087,752
var = 'e18500_capped'
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')


# SALT
# irs values  nrets
# irs 17in21id.xls taxes paid deduction 624,820,806  46,431,232
# irs 17in21id.xls income tax  368,654,631
# irs 17in21id.xls sales tax   20,734,779
# irs 17in21id.xls real estate 222,237,629
# irs 17in21id.xls personal property taxes 10,679,233
# irs 17in21id.xls other taxes 2,514,534

# puf values (2017)
# note: c21060 is Itemized deductions before phase-out (zero for non-itemizers)

# c18300 Sch A: State and local taxes plus real estate taxes deducted (component of pre-limitation c21060 total)

# e18400 Itemizable state and local income/sales taxes

# e18400_capped Sch A: State and local income taxes deductible, capped as a decimal fraction of AGI
# 526,195,784,967

# e18500 Itemizable real-estate taxes paid  285,719,544,931
# e18500_capped Sch A: State and local real estate taxes deductible, capped as a decimal fraction of AGI
# 285,719,544,931


# %% scratch interest paid deduction
irsvar = 'id_mortgage' # 292,557,787,000 33,746,351
irsvar = 'id_intpaid' #  313,944,112,000  34,327,403
print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

# e19200 Description: Itemizable interest paid
# c19200 Sch A: Interest deducted (component of pre-limitation c21060 total)
# e19200_capped Sch A: Interest deduction deductible, capped as a decimal fraction of AGI
var = 'e19200' #  424,406,109,267  55,333,072
var = 'c19200' # 357,486,840,616  36,146,781
var = 'e19200_capped'  # 424,406,109,267 55,333,072
print(f'{wsum(var):,.0f}')
print(f'{nret(var):,.0f}')
# seems like I should link c19200 to id_intpaid


# %% scratch charitable contributions

irsvar = 'id_contributions'
print(f'{irsn(irsvar):,.0f}')  # 37,979,015
print(f'{irssum(irsvar):,.0f}') # 256,064,685,000

# e19800 Itemizable charitable giving: cash/check contributions. WARNING: this variable is already capped in PUF data.
# e20100 Itemizable charitable giving: other than cash/check contributions. WARNING: this variable is already capped in PUF data.
# c19700 Sch A: Charity contributions deducted (component of pre-limitation c21060 total)
# e19800_capped Sch A: Charity cash contributions deductible, capped as a decimal fraction of AGI
# e20100_capped Sch A: Charity noncash contributions deductible, capped as a decimal fraction of AGI
var = 'e19800'  # 212,635,455,351   101,903,175
var = 'e20100'  # 64,207,135,577   56,359,659
var = 'c19700' # 211,099,226,362 38,613,998
var = 'e20100_capped' # 64,207,135,577  56,359,659
print(f'{wsum(var):,.0f}')
print(f'{nret(var):,.0f}')
# seems like we could use the sum of these e19800, e20100 as roughly equiv of id_contributions?
# for now match c19700 to id_contributions ??


# %% get nz counts and weighted sums of most puf variables, for FILERS
# get the subset of variables we want

# c18300 appears to be the SALT concept that corresponds to the uncapped deduction and comes a little
# close to what is in the irs spreadsheet
# c18300 Sch A: State and local taxes plus real estate taxes deducted (component of pre-limitation c21060 total)

keepcols = ('pid', 'common_stub', 's006', 'c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e01500', 'e02400', 'c02500',
            # itemized deductions
            'c17000', 'c18300', 'c19200', 'c19700')
pufsub = puf.loc[puf.filer, keepcols]

# make a long file with weighted values
puflong = pufsub.melt(id_vars=('pid', 'common_stub', 's006'))
puflong['nnz'] = (puflong.value != 0) * puflong.s006
puflong['wsum'] = puflong.value * puflong.s006

# get the sums by income range, add grand sums, add stub names
pufsums = puflong.groupby(['common_stub', 'variable'])[['nnz', 'wsum']].sum().reset_index()
grand_sums = pufsums.groupby(['variable']).sum().reset_index()
grand_sums['common_stub'] = 0
pufsums = pufsums.append(grand_sums)
pufsums = pd.merge(pufsums, pc.irsstubs, on=['common_stub'])
pufsums = pufsums.sort_values(['variable', 'common_stub'])
# reorder vars
vars = ['common_stub', 'incrange', 'variable', 'nnz', 'wsum']
pufsums = pufsums[vars]


# %% make long pufsums and map pufnames to irstot names
pufsumslong = pufsums.melt(id_vars=('common_stub', 'incrange', 'variable'), var_name='measure')
pufsumslong['puf_varmeas'] = pufsumslong.variable + '_' + pufsumslong.measure
pufsumslong.puf_varmeas.value_counts()

vmap = {# agi income components
        'c00100_nnz': 'nret_all',
        'c00100_wsum': 'agi',
        'e00200_nnz': 'nret_wages',
        'e00200_wsum': 'wages',
        'e00300_nnz': 'nret_taxint',
        'e00300_wsum': 'taxint',
        'e00600_nnz': 'nret_orddiv',
        'e00600_wsum': 'orddiv',
        'c01000_nnz': 'nret_cgnet',
        'c01000_wsum': 'cgnet',
        'e01500_nnz': 'nret_pensions',
        'e01500_wsum': 'pensions',
        'e02400_nnz': 'nret_socsectot',
        'e02400_wsum': 'socsectot',
        'c02500_nnz': 'nret_socsectaxable',
        'c02500_wsum': 'socsectaxable',
        # itemized deductions
        'c17000_nnz': 'nret_id_medical_capped',
        'c17000_wsum': 'id_medical_capped',
        'c18300_nnz': 'nret_id_taxpaid',
        'c18300_wsum': 'id_taxpaid',
        'c19200_nnz': 'nret_id_intpaid',
        'c19200_wsum': 'id_intpaid',
        'c19700_nnz': 'nret_id_contributions',
        'c19700_wsum': 'id_contributions'
        }

pufsumslong['irsvar'] = pufsumslong.puf_varmeas.map(vmap)
pufsumslong


# %% merge targets and pufsums, calc differences
irstot.info()
pufsumslong.info()
pufsumslong.puf_varmeas.value_counts()
comp = pd.merge(irstot.rename(columns={'variable': 'irsvar', 'value': 'irs'}),
                pufsumslong.rename(columns={'variable': 'pufvar', 'value': 'puf'}),
                on=['common_stub', 'incrange', 'irsvar'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
# reorder
mainvars = ['common_stub', 'incrange', 'irsvar', 'puf_varmeas',
            'irs', 'puf', 'diff', 'pdiff']
infovars = ['column_description', 'table_description', 'src']
comp = comp[mainvars + infovars]
comp.info()

comp.puf_varmeas.value_counts()


# %% print or write results

s = comp.copy()[mainvars + infovars]
# define custom sort order
s['puf_varmeas'] = pd.Categorical(s['puf_varmeas'], categories=vmap.keys())
s = s.sort_values(by=['puf_varmeas', 'common_stub'])

s['pdiff'] = s['pdiff'] / 100.0
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:.1%}'}
for key, value in format_mapping.items():
    s[key] = s[key].apply(value.format)

vlist = s.puf_varmeas.unique().tolist()
vlist

# for var in vlist:
#     print('\n\n')
#     s2 = s[s.puf_varmeas==var]
#     print(s2)

# pick one of the following 2 file names
# fname = r'C:\Users\donbo\Google Drive\NY PUF project\irs_puf_compare.txt'
fname = r'C:\programs_python\puf_analysis\results\irs_puf_compare.txt'

tfile = open(fname, 'a')
tfile.truncate(0)
# first write a summary with stub 0 for all variables
tfile.write('\n\n')
tfile.write('Summary report:\n')
tfile.write('  puf.csv advanced to 2017 with stage 1 and stage 2 only\n')
tfile.write('  filers only, using requirement rules and likely information\n\n')
s2 = s[s.common_stub==0]
tfile.write(s2.to_string())
# now write details for each variable
tfile.write('\n\nDetails by AGI range:')
for var in vlist:
    tfile.write('\n\n')
    s2 = s[s.puf_varmeas==var]
    tfile.write(s2.to_string())
tfile.close()


# %% OLD BELOW HERE develop usable targets

# drop targets for which I haven't yet set column descriptions as we won't
# use them
mask = irstot.variable.str.len() <= 2  # Excel column names will have length 2
irstot = irstot[~mask]
irstot = irstot.dropna(axis=0, subset=['column_description'])
irstot
irstot.columns

# check counts
irstot[['src', 'variable', 'table_description', 'value']].groupby(['src', 'table_description', 'variable']).agg(['count'])
vars = irstot[['variable', 'value']].groupby(['variable']).agg(['count'])  # unique list

# quick check to make sure duplicate variables have same values
# get unique combinations of src, variable
check = irstot[irstot.common_stub == 0][['src', 'variable']]
# indexes of duplicated combinations
idups = check.duplicated(subset='variable', keep=False)
check[idups].sort_values(['variable', 'src'])
dupvars = check[idups]['variable'].unique()
dupvars

# now check the stub 0 values of the variables that have duplicated values
qx = 'variable in @dupvars and common_stub==0'
vars = ['variable', 'column_description', 'src', 'value']
irstot.query(qx)[vars].sort_values(['variable', 'src'])
# looks ok except for very minor taxac differences
# any target version should be ok


# %% crosswalks on the fly
# dictionary xwalks between target name and puf name, AFTER constructing
# variables as needed in targets and in puf (as noted below)
TARGPUF_XWALK = dict(nret_all='nret_all',  # Table 1.1
                     # puf create nret_mars2, nret_mars1
                     nret_mfjss='nret_mars2',  # Table 1.2
                     nret_single='nret_mars1',  # Table 1.2
                     agi='c00100',  # Table 1.1
                     wages='e00200',  # Table 1.4
                     taxint='e00300',  # Table 1.4
                     orddiv='e00600',  # Table 1.4
                     # target cgnet = cggross - cgloss   # Table 1.4
                     cgnet='c01000',  # create cgnet in targets
                     # puf irapentot = e01400 + e01500 (taxable)
                     irapentot='irapentot',  # irapentot create in puf
                     socsectot='e02400',  # Table 1.4 NOTE THAT this is 'e'
                     ti='c04800'  # Table 1.1
                     )
TARGPUF_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
PUFTARG_XWALK = {val: kw for kw, val in TARGPUF_XWALK.items()}


# %% prepare targets for comparison based on xwalks above
# define target variables
tlist = list(TARGPUF_XWALK.keys())  # values we want
# compute  cgnet = cggross - cgloss   # Table 1.4
tlist.remove('cgnet')
tlist.append('cggross')
tlist.append('cgloss')
tlist

# get the proper data
irstot
qx1 = 'variable in @tlist and '
qx2 = '((variable not in @dupvars) or '
qx3 = '(variable in @dupvars and src=="17in11si.xls"))'
qx = qx1 + qx2 + qx3
qx
vars = ['variable', 'common_stub', 'value']
target_base = irstot.query(qx)[vars]
target_base[['variable', 'value']].groupby(['variable']).agg(['count'])
# good, this is what we want

wide = target_base.pivot(index='common_stub', columns='variable', values='value')
# multiply the dollar-valued columns by 1000 (i.e., the non-num cols)
numcols = ['nret_all', 'nret_mfjss', 'nret_single']
dollcols = np.setdiff1d(wide.columns, numcols)
dollcols
wide[dollcols] = wide[dollcols] * 1000
wide['cgnet'] = wide['cggross'] - wide['cgloss']
wide = wide.drop(['cggross', 'cgloss'], axis=1)
wide['common_stub'] = wide.index
wide.columns
targets_long = pd.melt(wide, id_vars=['common_stub'])
targets_long['variable'].value_counts()


# alternative: here is the numpy equivalent to R ifelse
# targets_long['value'] = np.where(condition, targets_long['value'] * 1000, targets_long['value'])



