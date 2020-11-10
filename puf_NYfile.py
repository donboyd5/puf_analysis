# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 04:18:49 2020

@author: donbo
"""

# %% notes
# Caution: keep show variables (in python console) off until needed as it slows things down


# %% imports
import sys
import taxcalc as tc
import pandas as pd
import numpy as np

from timeit import default_timer as timer
from datetime import date

import puf_constants as pc
import puf_utilities as pu

# microweight - this is sufficient
sys.path.append('c:/programs_python/weighting/')  # needed
import src.microweight as mw


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% functions
def uvals(series):
    return sorted(series.unique())


# %% get ht2_shares
ht2_shares = pd.read_csv(DATADIR + 'ht2_shares.csv')
ht2_shares
ht2_shares.info()
sts = uvals(ht2_shares.state)
len(sts)  # 52
" ,".join(sts)  # includes DC, OA, but not PR or US


# %% define states to target
compstates = ('NY', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX', 'VT')

# collapse target shares to these states and all others
m_states = ht2_shares.state.isin(compstates)
ht2_shares['stgroup'] = ht2_shares.state
ht2_shares.loc[~m_states, 'stgroup'] = 'other'
uvals(ht2_shares.stgroup)
aggvars = ['stgroup', 'pufvar', 'ht2var', 'ht2description', 'ht2_stub']
ht2_collapsed = ht2_shares.groupby(aggvars).agg({'share': 'sum', 'ht2': 'sum'}).reset_index()
ht2_collapsed.info()


# %% get relevant national puf
puf = pd.read_parquet(PUF_REGROWN)
puf


# %% add puf variables, get puf sums
puf2 = pu.prep_puf(puf, pufvars_to_nnz=None)
puf2 is puf # true!