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
RESULTDIR = r'C:\programs_python\puf_analysis\results/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'


# %% check
ht2_path = DATADIR + 'ht2_long.csv'

ht2 = pd.read_csv(ht2_path)  # 87,450
check = ht2[ht2.ht2var == 'n18500']

# put national value on every record
ht2_us = ht2[ht2["state"] == "US"] \
    [['ht2var', 'ht2_stub', 'ht2']].rename(columns={'ht2': 'ht2_us'})  # 1,650

# share are not perfectly 1 with national values, so let's use sum instead
ht2_sums = ht2[ht2["state"] != "US"][['ht2var', 'ht2_stub', 'ht2']]
ht2_sums = ht2_sums.groupby(['ht2var', 'ht2_stub'])['ht2'].sum()  # pandas treats na as zero
ht2_sums = ht2_sums.reset_index().rename(columns={'ht2': 'ht2_sum'})

ht2_shares = pd.merge(ht2[ht2.state != 'US'], ht2_sums, on=['ht2var', 'ht2_stub'])
ht2_shares['share'] = ht2_shares.ht2 / ht2_shares.ht2_sum

tmp = ht2_shares[ht2_shares.ht2_stub != 0].groupby(['ht2var', 'ht2_stub'])['share'].sum()
# note that some items are zero (e.g., vita_eic for high income)

ht2_shares.to_csv(DATADIR + 'ht2_shares.csv', index=None)


# %% functions
def calc_ratios(self):
    """
    This is based on Peter Metz's function
    For each target and AGI group, calculate the ratio of the state
    totals to the national totals using HT2
    """
    ht2 = pd.read_csv(self.ht2_path)

        # Filter for US
        ht2_us = ht2[ht2["STATE"] == "US"]

        keep_list = list(self.var_list)
        additional_vars = ["STATE", "AGI_STUB"]
        keep_list.extend(additional_vars)

        states = list(ht2.STATE.unique())
        # remove US from list of states
        states.pop(0)

        ht2_us_vals = ht2_us.drop(["STATE", "AGI_STUB"], axis=1)

        # Loop through each state to construct table of ratios
        ratio_df = pd.DataFrame()
        for state in states:
            state_df = ht2[ht2["STATE"] == state].reset_index()
            state_id = state_df[["STATE", "AGI_STUB"]]
            state_vals = state_df.drop(["index", "STATE", "AGI_STUB"], axis=1)

            # divide each state's total/stub by the U.S. total/stub
            ratios = state_vals / ht2_us_vals
            # tack back on states and stubs
            ratios_state = pd.concat([state_id, ratios], axis=1)
            # add each state ratio df to overall ratio df
            ratio_df = pd.concat([ratio_df, ratios_state])

        ratio_df = ratio_df[keep_list]
        return ratio_df