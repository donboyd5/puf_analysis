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
from datetime import date

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
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% get target data
targets_possible = pd.read_csv(DATADIR + 'targets2017_possible.csv')

target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
target_vars = target_mappings.pufvar.to_list()

# prepare puf
# get names of puf variables for which we will need to create nnz indicator
innz = target_mappings.pufvar.str.contains('_nnz')
nnz_vars = target_mappings.pufvar[innz]
pufvars_to_nnz = nnz_vars.str.rsplit(pat='_', n=1, expand=True)[0].to_list()


# %% get advanced regrown puf file
puf = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
puf.info()
puf.tail()

puf = pu.prep_puf(puf, pufvars_to_nnz)

# prepare puf subset
idvars = ['pid', 'filer', 'common_stub', 's006']
keep_vars = idvars + target_vars
pufsub = puf.loc[puf['filer'], keep_vars]
pufsub.columns.to_list()

# pufvars = puf.columns.sort_values().tolist()  # show all column names

# %% wide targets?
targets_possible.info()
possible_wide = targets_possible.loc[:, ['common_stub', 'pufvar', 'irs']] \
    .pivot(index='common_stub', columns='pufvar', values='irs') \
        .reset_index()

possible_wide.to_csv('c:/temp/wide.csv')

# %% define targets to use
target_vars
# targets_use = [target_vars[i] for i in [0, 2, 1, 4, 5, 6]]
# targets_use = [target_vars[i] for i in range(0, 8)]

# * is the unpacking operator
iuse = [*range(0, 8), *range(9, 27)]
iuse = [*range(0, 30)]
iuse
targets_use = [target_vars[i] for i in iuse]
targets_use = target_vars.copy()
# drop total social security as it does not seem reasonable??
targets_use.remove('e02400')
targets_use.remove('e02400_nnz')
targets_use


# %% run the stub
stub = 2

targets_use = target_vars.copy()
if stub in range(1, 5):
    # medical capped
    targets_use.remove('c17000')
    targets_use.remove('c17000_nnz')
    # contributions
    targets_use.remove('c19700')
    targets_use.remove('c19700_nnz')

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

opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'method': 'bvls', 'max_iter': 50}
# opts = None
probrw = prob.reweight(method='lsq', options=opts)
probrw.pdiff

# %% examine results
probrw.sspd
probrw.pdiff
np.quantile(probrw.g, qtiles)
probrw.elapsed_seconds
probrw.opts

x = probrw.g

# check
t1 = constraints(x, wh, xmat)
pdiff1 = t1 / targets_stub * 100 - 100
pdiff1



# %% function to do a single stub

def stub_opt(df):
    print(df.name)
    stub = df.name

    targets_use = target_vars.copy()
    if stub in range(1, 5):
        # medical capped
        targets_use.remove('c17000')
        targets_use.remove('c17000_nnz')
        # contributions
        targets_use.remove('c19700')
        targets_use.remove('c19700_nnz')

    pufrw = df.query('common_stub== @stub')[['pid', 's006'] + targets_use]
    targvals = possible_wide.loc[[stub], targets_use]

    xmat = np.asarray(pufrw[targets_use], dtype=float)
    wh = np.asarray(df.s006)
    targets_stub = np.asarray(targvals, dtype=float).flatten()

    x0 = np.ones(wh.size)

    prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)

    opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'method': 'bvls', 'max_iter': 50}
    probrw = prob.reweight(method='lsq', options=opts)

    df['x'] = probrw.g
    return df


# %% loop through puf
# target_vars
# # * is the unpacking operator
# iuse = [*range(0, 8), *range(9, 27)]
# iuse
# targets_use = [target_vars[i] for i in iuse]
# targest_use = target_vars
# targets_use
# tmp = possible_wide.loc[[18], targets_use]

grouped = pufsub.groupby('common_stub')

a = timer()
puf_rwtd = grouped.apply(stub_opt)
b = timer()
b - a

puf_rwtd
np.quantile(puf_rwtd.x, qtiles)

puf_rwtd['s006_rwt'] = puf_rwtd.s006 * puf_rwtd.x
puf_rwtd.info()

weights = puf_rwtd[['pid', 's006_rwt']]
weights.s006_rwt.sum()

pufsub.s006.sum()


# %% merge new weights back with original weights
puf_regrown_reweighted = puf.copy()
puf_regrown_reweighted['s006_taxcalc'] = puf_regrown_reweighted.s006
puf_regrown_reweighted = pd.merge(puf_regrown_reweighted, weights,
                                  on=['pid'], how='left')
rwmask = ~puf_regrown_reweighted.s006_rwt.isna()
rwmask.sum()
puf_regrown_reweighted.loc[rwmask, ['s006']] = puf_regrown_reweighted.s006_rwt[rwmask]

puf_regrown_reweighted[['pid', 's006', 's006_taxcalc', 's006_rwt']].tail(100)
puf_regrown_reweighted[['pid', 's006', 's006_taxcalc', 's006_rwt']].head(100)

puf_regrown_reweighted['s006'] = np.where(puf_regrown_reweighted.s006_rwt.isna(),
                                          puf_regrown_reweighted.s006,
                                          puf_regrown_reweighted.s006_rwt)
puf_regrown_reweighted.s006.sum()
puf_regrown_reweighted.s006_taxcalc.sum()
puf_regrown_reweighted.s006_rwt.sum()

puf_regrown_reweighted.to_parquet(PUFDIR + 'puf2017_regrown_reweighted.parquet', engine='pyarrow')


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

