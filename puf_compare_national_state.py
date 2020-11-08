# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 05:56:47 2020

@author: donbo
"""


# %% imports
import requests
import pandas as pd
import numpy as np
from io import StringIO
import json

import puf_constants as pc


# %% locations and file names
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
DOWNDIR = IGNOREDIR + 'downloads/'
HT2DIR = IGNOREDIR + 'Historical Table 2/'
DATADIR = 'C:/programs_python/puf_analysis/data/'


# %% get national and state files
# get previously determined national-puf mapping
# json.dump(pufirs_fullmap, open(DATADIR + 'pufirs_fullmap.json', 'w'))
pufirs_fullmap = json.load(open(DATADIR + 'pufirs_fullmap.json'))

targets_national = pd.read_csv(DATADIR + 'targets2017_possible.csv')
targets_ht2 = pd.read_csv(DATADIR + 'ht2_long.csv')

# targets_national.common_stub.value_counts().sort_values()
# targets_national.dtypes




# %% create mergeable files

# create common income range mapping between the national and state summary files
pc.ht2common_stubs  # map ht2stubs and common stubs to these stubs

# keys are the original values, values are the new values
# ht2 stubs are an easy mapping
keys = range(0, 11)
values = (0,1,1) + tuple(range(2, 10))
ht2_map = dict(zip(keys, values))

keys = range(0, 19)
values = (0, ) + (1,) * 2 + (2,) * 3 + (3,) * 3 + tuple(range(4, 9)) + (9,) * 5
irs_map = dict(zip(keys, values))
irs_map


# national drop vars, add ht2common_stub
targets_national.loc[:, ['ht2common_stub']] = targets_national.common_stub.map(irs_map)
aggcols = ['ht2common_stub', 'pufvar', 'irsvar', 'column_description', 'src', 'table_description', 'excel_column']
targets_national_mrg = targets_national.groupby(aggcols)[['irs']].sum().reset_index()
targets_national_mrg.columns


# ht2 keep US, add ht2common_stub
targets_ht2.loc[:, ['ht2common_stub']] = targets_ht2.ht2_stub.map(ht2_map)
targets_ht2 = targets_ht2[targets_ht2.state=='US'].drop(columns=['state'])
targets_ht2
aggcols = ['ht2common_stub', 'pufvar', 'ht2var', 'ht2description']
targets_ht2_mrg = targets_ht2.groupby(aggcols)[['ht2']].sum().reset_index()
targets_ht2_mrg.columns


# %% merge
natstate = pd.merge(targets_national_mrg, targets_ht2_mrg,
                    how='inner', on=['ht2common_stub', 'pufvar'])
natstate['diff'] = natstate.ht2 - natstate.irs
type(natstate)
natstate.columns
type(natstate.diff)
natstate['pdiff'] = natstate['diff'] / natstate.irs * 100
natstate['abspdiff'] = np.abs(natstate.pdiff)



