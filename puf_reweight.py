# coding: utf-8
"""
  # #!/usr/bin/env python
  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

  List of official puf files:
      https://docs.google.com/document/d/1tdo81DKSQVee13jzyJ52afd9oR68IwLpYZiXped_AbQ/edit?usp=sharing
      Per Peter latest file is here (8/20/2020 as of 9/13/2020)
      https://www.dropbox.com/s/hyhalpiczay98gz/puf.csv?dl=0
      C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20
      # raw string allows Windows-style slashes
      # r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

@author: donbo
"""

# %% imports
import sys
import taxcalc as tc
import pandas as pd
# import tables
import numpy as np

from bokeh.io import show, output_notebook
from timeit import default_timer as timer

# import src.reweight as rw
import puf_constants as pc
import puf_utilities as pu

# microweight - this is sufficient
sys.path.append('c:/programs_python/weighting/')  # needed
import src.microweight as mw


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


# %% locations and file names
# DATADIR = 'C:/programs_python/weighting/puf/data/'
# HDFDIR = 'C:/programs_python/weighting/puf/ignore/'

DATADIR = r'C:\programs_python\puf_analysis\data/'
HDFDIR = r'C:\programs_python\puf_analysis\ignore/'

# BASE_NAME = 'puf_adjusted'
BASE_NAME = 'puf2017_regrown2020-11-02'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)

# agi stubs
# AGI groups to target separately
# IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
#                  75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]

# HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
#                  200e3, 500e3, 1e6, 9e99]


# %% get target data and check them
# IRSDAT = DATADIR + 'targets2018.csv'
IRSDAT = DATADIR + 'targets2017_collapsed.csv'
irstot = pd.read_csv(IRSDAT)
# partnerscorploss
# TODO: fix all loss variables at source
irstot['value'] = np.where(irstot.variable=='partnerscorploss',
                           irstot.value * -1.0,
                           irstot.value)

# get irsstub and incrange mapping
# incrange for irsstub 0 and 1 doesn't have consistent text values so set them
# irstot.loc[irstot['irsstub'] == 0, 'incrange'] = 'All returns'
# irstot.loc[irstot['irsstub'] == 1, 'incrange'] = 'No adjusted gross income'

incmap = irstot[['common_stub', 'incrange']].drop_duplicates()
incmap

# drop targets for which I haven't yet set column descriptions as we won't
# use them
# irstot = irstot.dropna(axis=0, subset=['column_description'])
# irstot
irstot.columns

# check counts
irstot[['src', 'variable', 'value']].groupby(['src', 'variable']).agg(['count'])
irsvars = irstot[['variable', 'value']].groupby(['variable']).agg(['count'])  # unique list

# quick check to make sure duplicate variables have same values
# get unique combinations of src, variable
# check = irstot[irstot.common_stub == 0][['src', 'variable']]
# verify no duplicated combinations
# idups = irsvars.duplicated(subset='variable', keep=False)
# idups.sum()


# %% get advanced regrown puf file
# puf = pd.read_hdf(PUF_HDF)  # 1 sec
# puf.to_parquet(HDFDIR + BASE_NAME + '.parquet', engine='pyarrow')
# puf.to_feather(HDFDIR + BASE_NAME + '.feather')
puf = pd.read_parquet(HDFDIR + BASE_NAME + '.parquet', engine='pyarrow')
puf.info()
puf.tail()

puf['common_stub'] = pd.cut(
    puf['c00100'],
    pc.COMMON_STUBS,
    labels=range(1, 19),
    right=False)

puf['filer'] = pu.filers(puf, 2017)

pufvars = puf.columns.sort_values().tolist()  # show all column names


# %% select and create variables for reweighting
# when c04470 > 0 person is an itemizer
# item deds only for those with c04470 > 0 (i.e., itemizers)
# c00100, e00200, e00300, e00600, c01000, e01500, c02500, e26270pos, e26270neg,
# c17000, c18300, c19200, c19700

idvars = ['pid', 'common_stub', 'filer', 's006', 'MARS']
itemized = ['c04470']
incvars = ['c00100', 'e00200', 'e00300', 'e00600', 'c01000', 'e01500',
           'c02500', 'e26270']
dedvars = ['c17000', 'c18300', 'c19200', 'c19700']
keepvars = idvars + itemized + incvars + dedvars

# create puf extract that will be our base file for purposes of reweighting
pufx = puf[keepvars].copy()
pufx.columns.sort_values().tolist()

# transform as needed to create variables for reweighting
# we need nret_all
# we need mars indicators
# we need pos and neg values for some variables
# we need some only if itemizers

pufx['nret_all'] = 1

# marital status indicators
pufx['mars1'] = pufx.MARS.eq(1) * 1
pufx['mars2'] = pufx.MARS.eq(2) * 1
pufx['mars3'] = pufx.MARS.eq(3) * 1
pufx['mars4'] = pufx.MARS.eq(4) * 1
mvars = ['mars1', 'mars2', 'mars3', 'mars4']
# pufx['mars5'] = pufx.MARS.eq(5)

# partnership and S corp e26270
pufx['e26270pos'] = pufx.e26270 * pufx.e26270.gt(0)
pufx['e26270neg'] = pufx.e26270 * pufx.e26270.lt(0)
incvars_adj = [var for var in incvars if var != 'e26270'] \
    + ['e26270pos', 'e26270neg']

# create deduction vars that are positive only if record is an itemizer
for var in dedvars:
    pufx[var + 'id'] = pufx[var] * pufx.c04470.gt(0)
itemvars = [var + 'id' for var in dedvars]  # get list with names of these vars

# create indicators for number-of-nonzero variables
make_nzvars = incvars_adj + itemvars
for var in make_nzvars:
    pufx[var + '_nnz'] = pufx[var].ne(0) * 1  # pufx['s006'] * pufx[var].ne(0)
nzvars = [var + '_nnz' for var in make_nzvars]

rwvars = ['nret_all'] + mvars + incvars_adj + itemvars + nzvars
keepvars2 = idvars + rwvars

pufx.columns.sort_values().tolist()

# now create a file that has only filers and only the columns we want
pufsub = pufx.loc[pufx['filer'], keepvars2].copy()  # new data frame
pufsub.columns.sort_values().tolist()  # show all column names
pufsub.info()


# %% create crosswalk to rename irs vars to puf targeting vars

# filing status - just target the 3 biggest statuses
# they look close enough to work
# nret_all nret_all
# nret_single mars1
# nret_mfjss mars2
# nret_hoh mars4

# income items

# agi, c00100, target both nnz and amount
# wages, e00200, target both nnz and amount

# taxable interest income
# taxint irs line 8a, Form 1040, same as e00300
# DON'T TARGET NUMBER OF NZ RETURNS

# ordinary dividends line 9a form 1040
# orddiv same as e00600 the larger one
# looks like I should reduce the growfactor
# then should be ok to target both nnz and amount

# capital gains net income
# cgnet and c01000
# should be good to target both nnz and amount

# pensions total income
# pensions, e01500
# amount good, too many nnz in puf but PROB should target both

# social security taxable income
# socsectaxable, c02500
# looks like I should target both nnz and amount
# target taxable, not total

# partnership and scorp income and loss
# partnerscorpinc e26270pos
# partnerscorploss e26270neg
# should be able to target both nnz and amount for both

# itemized deductions
# irs id_medical_capped, c17000 the limited deduction in the irs data
# target both nnz and amount

# taxes paid deduction
# id_taxpaid, c18300  target both nnz and amount
# irs 17in21id.xls taxes paid deduction  46,431,232 624,820,806
# c18300 Sch A: State and local taxes plus real estate taxes deducted
#   (component of pre-limitation c21060 total)
# c18300  45,959,926 585,237,496,064
# wouldn't hurt to increase the growfactor

# interest paid deduction
# id_intpaid, c19200  should be able to target both nnz and amount

# charitable contributions deduction
# id_contributions c19700 appears correct


TARGPUF_XWALK = dict(
                     # total returns and marital status
                     nret_all='nret_all',  # Table 1.1
                     nret_mfjss='mars2',  # Table 1.2
                     nret_single='mars1',  # Table 1.2
                     nret_hoh='mars4',  # Table 1.2

                     # income items
                     agi='c00100',  # Table 1.1
                     wages='e00200',  # Table 1.4
                     nret_wages='e00200_nnz',  # Table 1.4
                     taxint='e00300',  # Table 1.4
                     nret_taxint='e00300_nnz',  # Table 1.4
                     orddiv='e00600',  # Table 1.4
                     nret_orddiv='e00600_nnz',  # Table 1.4
                     partnerscorpinc='e26270pos', # Table 1.4
                     nret_partnerscorpinc='e26270pos_nnz', # Table 1.4
                     partnerscorploss='e26270neg', # Table 1.4
                     nret_partnerscorploss='e26270neg_nnz', # Table 1.4
                     pensions='e01500',
                     nret_pensions='e01500_nnz',
                     socsectaxable='c02500',  # Table 1.4
                     nret_socsectaxable='c02500_nnz',  # Table 1.4
                     # target cgnet = cggross - cgloss   # Table 1.4
                     # cgnet='c01000',  # create cgnet in targets
                     # puf irapentot = e01400 + e01500 (taxable)
                     # irapentot='irapentot',  # irapentot create in puf
                     # socsectot='e02400',  # Table 1.4 NOTE THAT this is 'e'

                     # itemized deductions
                     id_medical_capped='c17000id',
                     nret_id_medical_capped='c17000id_nnz',
                     id_taxpaid='c18300id',
                     nret_id_taxpaid='c18300id_nnz',
                     id_intpaid='c19200id',
                     nret_id_intpaid='c19200id_nnz',
                     id_contributions='c19700id',
                     nret_id_contributions='c19700id_nnz'
                     )
TARGPUF_XWALK


# %% renames for irstot
irstot.columns
irstot['pufxvar'] = irstot.variable.map(TARGPUF_XWALK)
irstot.dropna(subset=['pufxvar'])[['variable', 'pufxvar', 'column_description']].drop_duplicates()

keep = ['common_stub', 'variable', 'pufxvar', 'column_description', 'value']
possible_targets = irstot.dropna(subset=['pufxvar'])[keep]


# %% prepare a puf summary for potential target variables
pufsub
targvars = list(TARGPUF_XWALK.values())
pufsub[targvars]

pufsums = pufsub.groupby('common_stub').apply(wsum,
                                              sumvars=targvars,
                                              wtvar='s006')

pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('common_stub')
pufsums = pufsums.reset_index()
pufsums

# pufsums = pufsums.rename(columns=PUFTARG_XWALK)
pufsums_long = pd.melt(pufsums, id_vars=['common_stub'])
pufsums_long


# %% check pufsums vs targets
possible_targets
pufsums_long

irscomp = possible_targets.rename(columns={'variable': 'irsvar', 'value': 'irs'})
irscomp
irscomp.info()

pufcomp = pufsums_long.rename(columns={'variable': 'pufxvar', 'value': 'puf'})
pufcomp
pufcomp.info()

comp = pd.merge(irscomp, pufcomp, on=['common_stub', 'pufxvar'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
comp['apdiff'] = np.abs(comp['pdiff'])
comp_s = comp.copy()
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:,.1f}',
                  'apdiff': '{:,.1f}'}
# caution: this changes numeric values to strings!
for key, value in format_mapping.items():
    comp_s[key] = comp_s[key].apply(value.format)
comp_s.info()
comp_s
tmp = comp_s.drop(columns=['column_description'])


# %% target an income range
targvars
targets_use = [targvars[i] for i in [0, 2, 1, 4, 5, 6]]
targets_use
target_base.pivot(index='irsstub', columns='variable', values='value')

possible_wide = possible_targets.pivot(index='common_stub', columns='pufxvar', values='value')

stub = 6
pufrw = pufsub.query('common_stub== @stub')[['pid', 's006'] + targets_use]
targvals = possible_wide.loc[[stub], targets_use]

xmat = np.asarray(pufrw[targets_use], dtype=float)
xmat.shape

wh = np.asarray(pufrw.s006)
targets_stub = np.asarray(targvals, dtype=float).flatten()

x0 = np.ones(wh.size)

# comp
t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets_stub * 100 - 100
pdiff0
np.square(pdiff0).sum()

prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)

opts = {'crange': 0.0001, 'xlb': 0, 'xub':1e5, 'quiet': False}
opts = {'crange': 0.001, 'xlb': 0, 'xub':50, 'quiet': False}
opts = None
rw1 = prob.reweight(method='ipopt', options=opts)
rw1.sspd
rw1.elapsed_seconds
rw1.pdiff
rw1.opts

# so = {'increment': 1e-3, 'autoscale': False}  # best 1819
# opts = {'increment': .00001}
# opts = {'increment': .00001, 'autoscale': False}
# opts = {'increment': 1e-6, 'autoscale': True}
# opts = {'increment': 1e-3, 'autoscale': False}
# opts = {'increment': 1e-6, 'autoscale': True, 'objective': 'QUADRATIC'}
opts = {'increment': 1e-4, 'autoscale': False}
opts = None
rw2 = prob.reweight(method='empcal', options=opts)
rw2.sspd
rw2.pdiff
rw2.opts

pdiff0
np.square(pdiff0).sum()

rw3 = prob.reweight(method='rake')
rw3 = prob.reweight(method='rake', options={'max_rake_iter': 20})
rw3.sspd

opts = {'xlb': 0.01, 'xub': 100, 'tol': 1e-8, 'max_iter': 150}
opts = {'xlb': 0.0, 'xub': 1e5, 'tol': 1e-7, 'max_iter': 100}
opts = {'xlb': 0, 'xub': 100, 'max_iter': 100}
opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'max_iter': 500}

# THIS IS IT BELOW
# This is important
opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'method': 'bvls', 'max_iter': 50}
opts = None
rw4 = prob.reweight(method='lsq', options=opts)
rw4.elapsed_seconds
rw4.sspd
rw4.pdiff
rw4.opts
np.quantile(rw4.g, qtiles)

# don't bother with minNLP
# rw5 = prob.reweight(method='minNLP')
# rw5.sspd
# rw5.opts


# distribution of g values
np.quantile(rw1.g, qtiles)
np.quantile(rw2.g, qtiles)
np.quantile(rw3.g, qtiles)  # HUGE g
np.quantile(rw4.g, qtiles)

# time
rw1.elapsed_seconds
rw2.elapsed_seconds
rw3.elapsed_seconds
rw4.elapsed_seconds

# sum of squared percentage differences
rw1.sspd
rw2.sspd
rw3.sspd
rw4.sspd

targets_stub
rw4.targets_opt
rw4.wh_opt

np.dot(xmat.T, wh)
np.dot(xmat.T, rw4.wh_opt)

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets_stub * 100 - 100
pdiff1



# %% OLD below here


# %% get just the variables we want and create new needed variables
# get list of target variables we need to have
plist = list(PUFTARG_XWALK.keys())
plist
# add names of variables we need for calculations or targeting
plist.append('pid')
plist.append('IRS_STUB')
plist.append('s006')
plist.append('MARS')
# remove names of variables we are going to create
plist.remove('nret_all')  # create as 1
plist.remove('nret_mars1')  # MARS==1
plist.remove('nret_mars2')  # MARS==2
# pension variables needed to calculate irapentot, and remove that name
plist.append('e01400')
plist.append('e01500')
plist.remove('irapentot')  # e01400 + e01500
plist  # these are the variables we need from the puf

# is everything from plist available in pufsub.columns?
[x for x in plist if x not in pufsub.columns]  # yes, all set

pufbase = pufsub[plist].copy()
# pufbase = pufbase.rename(columns={"s006": "nret_all"})
pufbase['nret_all'] = 1
pufbase['nret_mars1'] = (pufbase['MARS'] == 1).astype(int)
pufbase['nret_mars2'] = (pufbase['MARS'] == 2).astype(int)
# verify
pufbase[['MARS', 'pid']].groupby(['MARS']).agg(['count'])
pufbase.nret_mars1.sum()
pufbase.nret_mars2.sum()
# all good
pufbase['irapentot'] = pufbase['e01400'] + pufbase['e01500']
# drop variables no longer needed
pufbase = pufbase.drop(['MARS', 'e01400', 'e01500'], axis=1)
pufbase
pufbase.columns
# reorder columns
idvars = ['pid', 'IRS_STUB', 's006']
targvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
pufcols = idvars + targvars
pufcols
pufbase = pufbase[pufcols]
pufbase.columns




# %% crosswalks for reweight variables
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



# %% prepare potential targets based on xwalks above
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
qx3 = '(variable in @dupvars and src=="18in11si.xls"))'
qx = qx1 + qx2 + qx3
qx
vars = ['variable', 'irsstub', 'value']
target_base = irstot.query(qx)[vars]
target_base[['variable', 'value']].groupby(['variable']).agg(['count'])
# good, this is what we want

wide = target_base.pivot(index='irsstub', columns='variable', values='value')
# multiply the dollar-valued columns by 1000 (i.e., the non-num cols)
numcols = ['nret_all', 'nret_mfjss', 'nret_single']
dollcols = np.setdiff1d(wide.columns, numcols)
dollcols
wide[dollcols] = wide[dollcols] * 1000
wide['cgnet'] = wide['cggross'] - wide['cgloss']
wide = wide.drop(['cggross', 'cgloss'], axis=1)
wide['irsstub'] = wide.index
wide.columns
targets_long = pd.melt(wide, id_vars=['irsstub'])
targets_long['variable'].value_counts()

# alternative: here is the numpy equivalent to R ifelse
# targets_long['value'] = np.where(condition, targets_long['value'] * 1000, targets_long['value'])


# %% prepare a puf summary for potential target variables

pufsums = pufbase.groupby('IRS_STUB').apply(wsum,
                                            sumvars=targvars,
                                            wtvar='s006')

pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('IRS_STUB')
pufsums['irsstub'] = pufsums.index
pufsums

pufsums = pufsums.rename(columns=PUFTARG_XWALK)
pufsums_long = pd.melt(pufsums, id_vars=['irsstub'])
pufsums_long


# %% combine IRS totals and PUF totals and compare
targets_long
pufsums_long

irscomp = targets_long.rename(columns={'value': 'irs'})
irscomp
irscomp.info()

pufcomp = pufsums_long.rename(columns={'value': 'puf'})
pufcomp
pufcomp.info()

comp = pd.merge(irscomp, pufcomp, on=['irsstub', 'variable'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:,.1f}'}
# caution: this changes numeric values to strings!
for key, value in format_mapping.items():
    comp[key] = comp[key].apply(value.format)
comp.info()
comp

# comp.to_string(formatters=fmat)

# look at selected variables
# comp.query('variable == "nret_all"')


def f(var):
    print(comp[comp['variable'] == var])


voi = ['nret_all', 'agi', 'wages']
f('nret_all')
f(voi[1])
f(voi[2])


# %% OLD target an income range
pufbase

pufbase.head()
pufbase.info()
pufbase.IRS_STUB.count()
pufbase.IRS_STUB.value_counts()
# pufbase['IRS_STUB'].value_counts()

targets_long

# prepare all targets
targets_all = irscomp.pivot(index='irsstub', columns='variable', values='irs')
targets_all = targets_all.rename(columns=TARGPUF_XWALK)
targets_all['IRS_STUB'] = targets_all.index
targets_all.columns

# prepare data
targcols = ['nret_all', 'c00100', 'e00200']
targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400', 'c04800']

# 10 good targets
targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400']

stub = 10
# pufstub = pufbase.loc[pufbase['IRS_STUB'] ==  stub]
pufstub = pufbase.query('IRS_STUB == @stub')

xmat = np.asarray(pufstub[targcols], dtype=float)
xmat.shape

wh = np.asarray(pufstub.s006)
targets_all.loc[targets_all['IRS_STUB'] == stub]
targets_stub = targets_all[targcols].loc[targets_all['IRS_STUB'] == stub]
targets_stub = np.asarray(targets_stub, dtype=float).flatten()

x0 = np.ones(wh.size)

# comp
t0 = constraints(x0, wh, xmat)
pdiff0 = t0 / targets_stub * 100 - 100
pdiff0
np.square(pdiff0).sum()

prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)

opts = {'crange': 0.0001, 'xlb': 0, 'xub':1e5, 'quiet': False}
opts = {'crange': 0.001, 'xlb': 0, 'xub':50, 'quiet': False}
opts = None
rw1 = prob.reweight(method='ipopt', options=opts)
rw1.sspd
rw1.elapsed_seconds
rw1.pdiff
rw1.opts

# so = {'increment': 1e-3, 'autoscale': False}  # best 1819
# opts = {'increment': .00001}
# opts = {'increment': .00001, 'autoscale': False}
# opts = {'increment': 1e-6, 'autoscale': True}
# opts = {'increment': 1e-3, 'autoscale': False}
# opts = {'increment': 1e-6, 'autoscale': True, 'objective': 'QUADRATIC'}
opts = {'increment': 1e-4, 'autoscale': False}
opts = None
rw2 = prob.reweight(method='empcal', options=opts)
rw2.sspd
rw2.pdiff
rw2.opts

pdiff0
np.square(pdiff0).sum()

rw3 = prob.reweight(method='rake')
rw3 = prob.reweight(method='rake', options={'max_rake_iter': 20})
rw3.sspd

opts = {'xlb': 0.01, 'xub': 100, 'tol': 1e-8, 'max_iter': 150}
opts = {'xlb': 0.0, 'xub': 1e5, 'tol': 1e-7, 'max_iter': 100}
opts = {'xlb': 0, 'xub': 100, 'max_iter': 100}
opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'max_iter': 500}

# THIS IS IT BELOW
# This is important
opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'method': 'bvls', 'max_iter': 50}
opts = None
rw4 = prob.reweight(method='lsq', options=opts)
rw4.elapsed_seconds
rw4.sspd
rw4.pdiff
rw4.opts
np.quantile(rw4.g, qtiles)

# don't bother with minNLP
# rw5 = prob.reweight(method='minNLP')
# rw5.sspd
# rw5.opts


# distribution of g values
np.quantile(rw1.g, qtiles)
np.quantile(rw2.g, qtiles)
np.quantile(rw3.g, qtiles)  # HUGE g
np.quantile(rw4.g, qtiles)

# time
rw1.elapsed_seconds
rw2.elapsed_seconds
rw3.elapsed_seconds
rw4.elapsed_seconds

# sum of squared percentage differences
rw1.sspd
rw2.sspd
rw3.sspd
rw4.sspd

targets_stub
rw4.targets_opt
rw4.wh_opt

np.dot(xmat.T, wh)
np.dot(xmat.T, rw4.wh_opt)

np.quantile(x, [0, .1, .25, .5, .75, .9, 1])

t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets_stub * 100 - 100
pdiff1


# %% loop through puf

def stub_opt(df):
    print(df.name)
    stub = df.name
    # pufstub = pufbase.loc[pufbase['IRS_STUB'] ==  stub]
    xmat = np.asarray(df[targcols], dtype=float)
    wh = np.asarray(df.s006)

    targets_all.loc[targets_all['IRS_STUB'] == stub]
    targets_stub = targets_all[targcols].loc[targets_all['IRS_STUB'] == stub]
    targets_stub = np.asarray(targets_stub, dtype=float).flatten()

    x0 = np.ones(wh.size)

    prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)

    opts = None
    rw = prob.reweight(method='lsq', options=opts)

    # rwp = rw.Reweight(wh, xmat, targets_stub)
    # x, info = rwp.reweight(xlb=0.1, xub=10,
    #                        crange=.0001,
    #                        ccgoal=10, objgoal=100,
    #                        max_iter=50)
    # print(info['status_msg'])

    df['x'] = rw.g
    return df


# targcols = ['nret_all', 'c00100', 'e00200']
alltargs = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400', 'c04800']

targcols = ['nret_all', 'nret_mars2', 'nret_mars1',
            'c00100', 'e00200', 'e00300', 'e00600',
            'irapentot', 'c01000', 'e02400']

grouped = pufbase.groupby('IRS_STUB')
temp = pufbase.loc[pufbase['IRS_STUB'] == 1]

a = timer()
dfnew = grouped.apply(stub_opt)
# dfnew = temp.groupby('IRS_STUB').apply(lambda x: stub_opt(x, targcols))
b = timer()
b - a

dfnew
dfnew['wtnew'] = dfnew.s006 * dfnew.x
dfnew.info()
# convert IRS_STUB from category to equivalent integer so that we can
# save as hdf5 format, which does not allow category variables
dfnew['IRS_STUB'] = dfnew['IRS_STUB'].cat.codes + 1
dfnew.info()


# %% save file
PUF_RWTD = HDFDIR + 'puf2018_reweighted_2020-10-23' + '.h5'
PUF_RWTD_CSV = HDFDIR + 'puf2018_reweighted_2020-10-23' + '.csv'

dfnew.to_hdf(PUF_RWTD, 'data')  # 1 sec
dfnew.to_csv(PUF_RWTD_CSV, index=None)  # a few secs

# get file
dfnew = pd.read_hdf(PUF_RWTD)  # 235 ms
dfcsv = pd.read_csv(PUF_RWTD_CSV)  # 235 ms



# %% examine results
targets_long
pufsums_long

result_sums = dfnew.groupby('IRS_STUB').apply(wsum,
                                               sumvars=alltargs,
                                               wtvar='wtnew')

result_sums = result_sums.append(result_sums.sum().rename(0)).sort_values('IRS_STUB')
result_sums
result_sums['irsstub'] = result_sums.index
result_sums = result_sums.rename(columns=PUFTARG_XWALK)
result_sums.columns

resultsums_long = pd.melt(result_sums, id_vars=['irsstub'])

resultscomp = resultsums_long
resultscomp = resultscomp.rename(columns={'value': 'pufrw'})
resultscomp
resultscomp.info()

# combine and format
comp2 = pd.merge(incmap, irscomp, on='irsstub')
comp2 = pd.merge(comp2, pufcomp, on=['irsstub', 'variable'])
comp2 = pd.merge(comp2, resultscomp, on=['irsstub', 'variable'])
comp2['puf_diff'] = comp2['puf'] - comp2['irs']
comp2['pufrw_diff'] = comp2['pufrw'] - comp2['irs']
comp2['puf_pdiff'] = comp2['puf_diff'] / comp2['irs'] * 100
comp2['pufrw_pdiff'] = comp2['pufrw_diff'] / comp2['irs'] * 100

# format the data - will change numerics to strings
# mcols = list(comp2.columns)
# mcols[2:7]
comp3 = comp2.copy()
condition = np.logical_not(comp3['variable'].isin(['nret_all', 'nret_mfjss', 'nret_single']))
condition.sum()
changevars = ['irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff']
# put dollar-valued items in $ millions
for var in changevars:
    comp3[var] = np.where(condition, comp3[var] / 1e9, comp3[var])

# CAUTION: This formatting creates strings!
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'pufrw': '{:,.0f}',
                  'puf_diff': '{:,.1f}',
                  'pufrw_diff': '{:,.1f}',
                  'puf_pdiff': '{:,.1f}',
                  'pufrw_pdiff': '{:,.1f}'}

for key, value in format_mapping.items():
    comp3[key] = comp3[key].apply(value.format)

dollarvars = ['irsstub', 'incrange', 'variable',
              'irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff']

pctvars = ['irsstub', 'incrange', 'variable',
              'irs', 'puf', 'pufrw', 'puf_pdiff', 'pufrw_pdiff']

allvars = ['irsstub', 'incrange', 'variable',
           'irs', 'puf', 'pufrw', 'puf_diff', 'pufrw_diff',
           'puf_pdiff', 'pufrw_pdiff']


# pd.options.display.max_columns = 8
# pd.reset_option('display.max_columns')
comp3['variable'].value_counts()
var = 'nret_all'
var = 'nret_mfjss'
var = 'nret_single'
var = 'agi'
var = 'irapentot'
var = 'taxint'
var = 'cgnet'
var = 'ti'
comp3.loc[comp3['variable'] == var, pctvars]
comp3.loc[comp3['variable'] == var, dollarvars]
# comp3.loc[comp3['variable'] == var, allvars]


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

