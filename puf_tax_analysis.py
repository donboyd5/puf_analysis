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

https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/REFORMS.md#how-to-specify-a-tax-reform-in-a-json-policy-reform-file


https://github.com/PSLmodels/Tax-Calculator/blob/master/docs/guide/policy_params.md

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

# reforms
law2017 = REFORMSDIR + '2017_law.json'
law2017_SALTcapped = REFORMSDIR + 'law2017_SALTcapped.json'  # relative to 2017 law!

law2018 = REFORMSDIR + 'TCJA.json'
law2018_SALTuncapped = REFORMSDIR + 'law2018_SALTuncapped.json'  # must be run relative to 2018+ law


# %% get reforms
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/2017_law.json
params2017 = tc.Calculator.read_json_param_objects(law2017, None)
params2017_SALTcapped = tc.Calculator.read_json_param_objects(law2017_SALTcapped, None)

params2018 = tc.Calculator.read_json_param_objects(law2018, None)
params2018_SALTuncapped = tc.Calculator.read_json_param_objects(law2018_SALTuncapped, None)


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



recs = tc.Records(data=puf2018,
                  start_year=2018,
                  weights=weights_us,
                  adjust_ratios=None)

 # note that we don't need to advance because start year is 2018


# %% 2018 law, 2018 data
pol = tc.Policy()
calc2018 = tc.Calculator(policy=pol, records=recs)  # current-law
calc2018.calc_all()  #
tax2018 = calc2018.weighted_total('iitax')
tax2018


# %% 2017 law, 2018 data
# now implement reforms
pol = tc.Policy()
pol.implement_reform(params2017['policy'])
calc2017 = tc.Calculator(policy=pol, records=recs)
calc2017.calc_all()
tax2017 = calc2017.weighted_total('iitax')
tax2017


# %% 2018 law with SALT uncapped, 2018 data
pol = tc.Policy()
pol.implement_reform(params2018_SALTuncapped['policy'])
calc2018xSALT = tc.Calculator(policy=pol, records=recs)
calc2018xSALT.calc_all()
tax2018xSALT = calc2018xSALT.weighted_total('iitax')
tax2018xSALT


# %% 2017 law with SALT added, 2018 data
pol = tc.Policy()
pol.implement_reform(params2017['policy'])
pol.implement_reform(params2017_SALTcapped['policy'])
calc2017_SALTcapped = tc.Calculator(policy=pol, records=recs)
calc2017_SALTcapped.calc_all()
tax2017SALTcapped = calc2017_SALTcapped.weighted_total('iitax')
tax2017SALTcapped


# %% comparisons below here
(tax2018 - tax2017) / 1e9
(tax2018 - tax2018xSALT) / 1e9
(tax2017SALTcapped - tax2017) / 1e9



calc2018.weighted_total('iitax')
calc2017.weighted_total('iitax')

tcut = calc2018.weighted_total('iitax') - calc2017.weighted_total('iitax')
tcut / 1e9
tcut / calc2017.weighted_total('iitax') * 100

# caution: policies build on each other; if we want to go back to default, reset the policy
pol = tc.Policy()
pol.implement_reform(params2018_SALTuncapped['policy'])
calc2018xSALT = tc.Calculator(policy=pol, records=recs)
calc2018xSALT.calc_all()
calc2018xSALT.weighted_total('iitax')

(calc2018xSALT.weighted_total('iitax') - calc2018.weighted_total('iitax')) / 1e9


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


# %% analyze

(puf2018.iitax * puf2018.s006).sum()

# run 2017 law and 2018 law on the file
# sum iitax using s006 weights on file 1509996924390.8943
(puf2018.iitax * puf2018.s006).sum()  #  1509996924390.8943
(puf2018.iitax * weights_us.WT2018 / 100).sum()  # 1509996924390.8943
# calc1.weighted_total('iitax') # 1509781471549.1584
(puf2018.iitax * weights_us.WT2018.astype('int32') / 100).sum()  # 1509781471549.1584
# those weights are same as weights in weights_us



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

