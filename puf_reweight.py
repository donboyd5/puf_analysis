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
from pathlib import Path
WEIGHTING_DIR = str(Path.home() / 'Documents/python_projects/weighting')
if WEIGHTING_DIR not in sys.path:
    sys.path.append(str(WEIGHTING_DIR))

import src.microweight as mw


# %% ONETIME create file of national weights
# puf.columns
# vars = ['pid', 's006_taxcalc', 's006_rwt', 's006_rwt_geo']
# national_weights = puf.loc[:, vars]
# national_weights.to_csv(PUFDIR + 'national_weights.csv', index=False)
# national_weights.isnull().sum()


# %% program functions

def dfvars(df):
    return df.columns.sort_values().tolist()

def constraints(x, wh, xmat):
    return np.dot(x * wh, xmat)


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'
PUF_REGROWN_RW_GW = PUFDIR + 'puf2017_regrown_reweighted_geoweighted.parquet'


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


# %% wide targets?
targets_possible.info()
possible_wide = targets_possible.loc[:, ['common_stub', 'pufvar', 'irs']] \
    .pivot(index='common_stub', columns='pufvar', values='irs') \
        .reset_index()
# possible_wide.to_csv('c:/temp/wide.csv')


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


# %% get desired puf and weights
# PUF_REGROWN  PUF_REGROWN_RW_GW
# puf = pd.read_parquet(PUF_REGROWN_RW_GW, engine='pyarrow')

puf = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
wts = pd.read_csv(PUFDIR + 'national_weights.csv')

dfvars(puf)  # has s006
# wts['weight'] = np.where(wts.s006_rwt_geo.isnan(),
#                          wts.s006_taxcalc,
#                          wts.s006_rwt_geo)

wts['weight'] = wts.s006_rwt_geo
wts['weight'].fillna(wts['s006_taxcalc'], inplace=True)

# put the desired weight on puf
puf = pd.merge(puf, wts[['pid', 'weight']], how='left')
puf.describe()  # make sure we have no missing weights


puf.info()
puf.tail()

puf = pu.prep_puf(puf, pufvars_to_nnz)

# prepare puf subset
idvars = ['pid', 'filer', 'common_stub', 'weight']
keep_vars = idvars + target_vars
pufsub = puf.loc[puf['filer'], keep_vars]
pufsub.columns.to_list()

# pufvars = puf.columns.sort_values().tolist()  # show all column names


# %% prep all for loop



# %% function to do a single stub
def stub_opt(df, method):
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

    pufrw = df.query('common_stub== @stub')[['pid', 'weight'] + targets_use]
    targvals = possible_wide.loc[[stub], targets_use]

    xmat = np.asarray(pufrw[targets_use], dtype=float)
    wh = np.asarray(df.weight)
    targets_stub = np.asarray(targvals, dtype=float).flatten()

    prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)

    if method == 'lsq':
        opts = {'xlb': 0.1, 'xub': 100, 'tol': 1e-7, 'method': 'bvls',
                'max_iter': 50}
    elif method == 'ipopt':
        # opts = {'crange': 0.001, 'quiet': False}
        opts = {'crange': 0.001, 'xlb': 0.1, 'xub': 100, 'quiet': False}

    rw = prob.reweight(method=method, options=opts)

    df['x'] = rw.g
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
puf_rwtd = grouped.apply(stub_opt, method='lsq')  # method lsq or ipopt
b = timer()
b - a

puf_rwtd
puf_rwtd.x.size
np.quantile(puf_rwtd.x, qtiles)

puf_rwtd.x.head(15)

# CAUTION: make sure to get the right weight to be multiplied by x
weights = puf_rwtd.loc[:, ['pid', 'weight', 'x']]
weights['weight_new'] = weights.weight * weights.x



# %% update and save our file of national weights
national_weights = pd.read_csv(PUFDIR + 'national_weights.csv')
weights = weights[['pid', 'weight_new']].rename(columns={'weight_new': 's006_rwt_geo_rwt'})
national_weights = pd.merge(national_weights.drop(columns='s006_rwt_geo_rwt'), weights, on='pid', how='left')
national_weights.describe()
national_weights.isna().sum()
national_weights.isnull().sum()
# DO NOT SAVE UNTIL CERTAIN THAT THE FILE IS GOOD
national_weights.to_csv(PUFDIR + 'national_weights.csv', index=False)


# %% SCRATCH area below here


# %% run a stub
stub = 1

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

opts = {'crange': 0.001, 'quiet': False}
# opts = {'crange': 0.001, 'xlb':0, 'xub':100, 'quiet': False}
rw1 = prob.reweight(method='ipopt', options=opts)
rw1.g.shape


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

stub = 1
df = pufsub.query('common_stub == @stub')




# %% OLD merge new weights back with original weights
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

