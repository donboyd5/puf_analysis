# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:57:20 2020

    Notes
    -----
    The most efficient way to specify current-law and reform Calculator
    objects is as follows:
         pol = Policy()
         rec = Records()
         calc1 = Calculator(policy=pol, records=rec)  # current-law
         pol.implement_reform(...)
         calc2 = Calculator(policy=pol, records=rec)  # reform
    All calculations are done on the internal copies of the Policy and
    Records objects passed to each of the two Calculator constructors.

Useful:
    https://pslmodels.github.io/Tax-Calculator/recipes/recipe00.html


"""


# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import puf_constants as pc
import puf_utilities as pu

from timeit import default_timer as timer
from importlib import reload


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
DATADIR = r'C:\programs_python\puf_analysis\data/'
REFORMSDIR = r'C:\programs_python\puf_analysis\reforms/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'

TCOUTDIR = IGNOREDIR + 'taxcalc_output/'


# %% constants
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'
law_2017 = REFORMSDIR + '2017_law.json'


# %% get reforms
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/2017_law.json
params_2017 = tc.Calculator.read_json_param_objects(law_2017, None)


# %% get data
puf = pd.read_csv(LATEST_OFFICIAL_PUF)

puf2018 = pd.read_parquet(PUFDIR + 'puf2018_weighted' + '.parquet', engine='pyarrow')
puf2018.c00100.describe()

sweights_2018 = pd.read_csv(PUFDIR + 'allweights2018_geo2017_grown.csv')

# check the weights
# puf2018.loc[puf2018.pid==11, ['pid', 's006']]
puf2018[['pid', 's006']].head(20)
sweights_2018.head(5)

# create a national weights dataframe suitable for tax-calculator
weights_us = sweights_2018.loc[:, ['pid', 'weight']].rename(columns={'weight': 'WT2018'})
weights_us = puf2018[['pid', 's006']].rename(columns={'s006': 'WT2018'})
weights_us['WT2018'] = weights_us.WT2018 * 100


# %% estimate reform impacts using puf.csv
recs_puf = tc.Records(data=puf)
pol_puf = tc.Policy()
calc_puf = tc.Calculator(policy=pol_puf, records=recs_puf)
calc_puf.advance_to_year(2018)
calc_puf.calc_all()
itax_puf = calc_puf.weighted_total('iitax')

pol_puf.implement_reform(params_2017['policy'])
calc_puf_2017 = tc.Calculator(policy=pol_puf, records=recs_puf)
calc_puf_2017.advance_to_year(2018)
calc_puf_2017.calc_all()
itax_puf_2017 = calc_puf_2017.weighted_total('iitax')

(itax_puf - itax_puf_2017) / 1e9
itax_puf / itax_puf_2017 * 100 - 100


# %% estimate same reform impacts using regrown reweighted puf
(puf2018.iitax * puf2018.s006).sum()

# run 2017 law and 2018 law on the file
# sum iitax using s006 weights on file 1509996924390.8943
(puf2018.iitax * puf2018.s006).sum()  #  1509996924390.8943
(puf2018.iitax * weights_us.WT2018 / 100).sum()  # 1509996924390.8943
# calc1.weighted_total('iitax') # 1509781471549.1584
(puf2018.iitax * weights_us.WT2018.astype('int32') / 100).sum()  # 1509781471549.1584
# those weights are same as weights in weights_us

recs = tc.Records(data=puf2018,
                  start_year=2018,
                  weights=weights_us,
                  adjust_ratios=None)

pol = tc.Policy()
calc_baseline = tc.Calculator(policy=pol, records=recs)  # current-law
# don't need to advance because start year is 2018
calc_baseline.calc_all()

pol.implement_reform(params_2017['policy'])
calc_2017 = tc.Calculator(policy=pol, records=recs)
calc_2017.calc_all()

itax_baseline = calc_baseline.weighted_total('iitax')
itax_2017 = calc_2017.weighted_total('iitax')

itax_puf
itax_puf_2017


itax_baseline / itax_2017 * 100 - 100
(itax_baseline - itax_2017) / 1e9


calc_baseline.weighted_total('iitax')

# %% save results

def f(tcout):
    df = tcout.dataframe(variable_list=[], all_vars=True)
    df['pid'] = puf2018.pid

    df['ht2_stub'] = pd.cut(
        df['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)
    # avoid categorical variable, it causes problems!
    df['ht2_stub'] = df.ht2_stub.astype('int64')


    df = pd.merge(df,
                  pc.ht2stubs.rename(columns={'ht2stub': 'ht2_stub'}),
                  how='left', on='ht2_stub')

    df['filer'] = puf2018.filer
    return df

df = f(calc_baseline)
df.to_parquet(TCOUTDIR + 'tcout2018_2018law.parquet', engine='pyarrow')

df = f(calc_2017)
df.to_parquet(TCOUTDIR + 'tcout2018_2017law.parquet', engine='pyarrow')

dfbl = pd.read_parquet(TCOUTDIR + 'tcout2018_2018law.parquet', engine='pyarrow')
dfbl.to_csv(TCOUTDIR + 'tcout2018_2018law.csv', index=None)

df2017 = pd.read_parquet(TCOUTDIR + 'tcout2018_2017law.parquet', engine='pyarrow')
df2017.to_csv(TCOUTDIR + 'tcout2018_2017law.csv', index=None)


