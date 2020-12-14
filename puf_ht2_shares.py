# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 06:52:22 2020

@author: donbo
"""

# %% imports
import pandas as pd
import numpy as np
import puf_constants as pc
from datetime import date
import json

import puf_utilities as pu


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
RESULTDIR = r'C:\programs_python\puf_analysis\result_tables/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'


# %% check
ht2_path = DATADIR + 'ht2_long.csv'

ht2 = pd.read_csv(ht2_path)  # 87,450
check = ht2[ht2.ht2var == 'n18500']
# temp = ht2[['ht2var', 'ht2description', 'pufvar']].drop_duplicates()
pu.uvals(ht2.ht2var)

# put national value on every record
ht2_us = ht2[ht2["state"] == "US"] \
    [['ht2var', 'ht2_stub', 'ht2']].rename(columns={'ht2': 'ht2_us'})  # 1,650

# shares are not perfectly 1 with national values, so let's use sum instead
ht2_sums = ht2[ht2["state"] != "US"][['ht2var', 'ht2_stub', 'ht2']]
ht2_sums = ht2_sums.groupby(['ht2var', 'ht2_stub'])['ht2'].sum()  # pandas treats na as zero
ht2_sums = ht2_sums.reset_index().rename(columns={'ht2': 'ht2_sum'})

ht2_shares = pd.merge(ht2[ht2.state != 'US'], ht2_sums, on=['ht2var', 'ht2_stub'])
ht2_shares['share'] = ht2_shares.ht2 / ht2_shares.ht2_sum

tmp = ht2_shares[ht2_shares.ht2_stub != 0].groupby(['ht2var', 'ht2_stub'])['share'].sum()
# note that some items are zero (e.g., vita_eic for high income)

ht2_shares.to_csv(DATADIR + 'ht2_shares.csv', index=None)
# pu.uvals(ht2_shares.ht2var)

ht2_shares.query('state=="NY" & ht2var=="aiitax"')


