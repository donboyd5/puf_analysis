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

switch:
    @donboyd5 It has certainly helped to clarify my thinking!
    By the way, do you know that you can check out this branch locally?
    git fetch upstream pull/2497/head:pr-2497

    https://github.com/PSLmodels/Tax-Calculator/pull/2497

When modifying tax calculator source:
	from Tax-Calculator directory

	after revision of source or checkout of new version:
	if there is an old one in place:
		pip uninstall taxcalc
	then:
		python setup.py install

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

REFDIR = r'C:\programs_python\puf_analysis\reforms/'

IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'

TCOUTDIR = IGNOREDIR + 'taxcalc_output/'


# %% constants
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'  # August 20, 2020 puf.csv


# %% reforms -- file names or dicts
law2017 = tc.Policy.read_json_reform(REFDIR + '2017_law.json')

# reforms needed to walk from 2017 to 2018 law
# Note: # tc.Policy().read_json_reform(qbid_limit) -- creates a dict out of
# a json file, converting false or "false" to False -- so I can create dicts directly
salt2018 = {"ID_AllTaxes_c": {"2018": [10000.0, 10000.0, 5000.0, 10000.0, 10000.0]}}

sd2018 = {"STD": {"2018": [12000, 24000, 12000, 18000, 24000]}}

persx2018 = {"II_em": {"2018": 0}}

rates2018 = {"II_rt1": {"2018": 0.10},
             "II_rt2": {"2018": 0.12},
             "II_rt3": {"2018": 0.22},
             "II_rt4": {"2018": 0.24},
             "II_rt5": {"2018": 0.32},
             "II_rt6": {"2018": 0.35},
             "II_rt7": {"2018": 0.37},
             "II_brk1": {"2018": [9525, 19050, 9525, 13600, 19050]},
             "II_brk2": {"2018": [38700, 77400, 38700, 51800, 77400]},
             "II_brk3": {"2018": [82500, 165000, 82500, 82500, 165000]},
             "II_brk4": {"2018": [157500, 315000, 157500, 157500, 315000]},
             "II_brk5": {"2018": [200000, 400000, 200000, 200000, 400000]},
             "II_brk6": {"2018": [500000, 600000, 300000, 500000, 600000]}}

# wondering if I should treat other pass-through provisions together with this
ptrates2018 = {"PT_rt1": {"2018": 0.10},
               "PT_rt2": {"2018": 0.12},
               "PT_rt3": {"2018": 0.22},
               "PT_rt4": {"2018": 0.24},
               "PT_rt5": {"2018": 0.32},
               "PT_rt6": {"2018": 0.35},
               "PT_rt7": {"2018": 0.37},
               "PT_brk1": {"2018": [9525, 19050, 9525, 13600, 19050]},
               "PT_brk2": {"2018": [38700, 77400, 38700, 51800, 77400]},
               "PT_brk3": {"2018": [82500, 165000, 82500, 82500, 165000]},
               "PT_brk4": {"2018": [157500, 315000, 157500, 157500, 315000]},
               "PT_brk5": {"2018": [200000, 400000, 200000, 200000, 400000]},
               "PT_brk6": {"2018": [500000, 600000, 300000, 500000, 600000]}}

allrates2018 = {**rates2018, **ptrates2018}

qbid2018 = {"PT_qbid_rt": {"2018": 0.2},
            "PT_qbid_taxinc_thd": {"2018": [157500, 315000, 157500, 157500, 315000]},
            "PT_qbid_taxinc_gap": {"2018": [50000, 100000, 50000, 50000, 100000]},
            "PT_qbid_w2_wages_rt": {"2018": 0.5},
            "PT_qbid_alt_w2_wages_rt": {"2018": 0.25},
            "PT_qbid_alt_property_rt": {"2018": 0.025}}
qbid_limitfalse = {"PT_qbid_limit_switch": {"2018": False}}
qbid2018_limitfalse = {**qbid2018, **qbid_limitfalse}

passthrough_qbidxlimit2018 = {**ptrates2018, **qbid2018_limitfalse}

amt2018 ={"AMT_em": {"2018": [70300, 109400, 54700, 70300, 109400]},
          "AMT_em_ps": {"2018": [500000, 1000000, 500000, 500000, 1000000]},
          "AMT_em_pe": {"2018": 718800}}

law2018 = tc.Policy.read_json_reform(REFDIR + 'TCJA.json')

law2018xQlimit = {**law2018, **qbid_limitfalse}  # TCJA but with qbid limit set to false

# caution: this next reform is based on carefully deleting provisions already estimated
# above. needs to be checked/reviewed
other_2018vs2017 = tc.Policy.read_json_reform(REFDIR + 'other_2018vs2017.json')


# %% functions to add reform, save tc output
def add_reform(reform_name):
    global order  # we will modify this
    global wsum_prior

    pol.implement_reform(eval(reform_name))
    calc = tc.Calculator(policy=pol, records=recs)
    calc.calc_all()
    wsum = calc.weighted_total('iitax') / 1e9
    wsum_change = wsum - wsum_prior
    print(f'{reform_name:<30} {wsum:>9,.0f}  {wsum_change:>9,.1f}  iitax total and change in $ billions')

    # note that prep_tcout needs pidfiler and puf_constants in global env
    df = prep_tcout(calc, reform_name, order)
    output_name = STACKDIR + str(order) + '_' + reform_name + '.parquet'
    df.to_parquet(output_name, engine='pyarrow')

    order = order + 1
    wsum_prior = wsum
    return None


def prep_tcout(tcout, reform_name=None, order=None):
    # note: pidfiler must exist in the global environment
    df = tcout.dataframe(variable_list=[], all_vars=True)
    df['pid'] = pidfiler.pid
    df['filer'] = pidfiler.filer

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
    return df


def solo_reform(reform_name, base_name, calc_base):
    pol = tc.Policy()
    pol.implement_reform(eval(base_name))
    pol.implement_reform(eval(reform_name))
    calc = tc.Calculator(policy=pol, records=recs)
    calc.calc_all()
    wsum = calc.weighted_total('iitax') / 1e9
    wsum_base = calc_base.weighted_total('iitax') / 1e9
    wsum_change = wsum - wsum_base
    print(f'{reform_name:<30} vs. {base_name} {wsum:>9,.0f}  {wsum_base:>9,.0f}  {wsum_change:>9,.1f}  iitax total and change in $ billions')

    # note that prep_tcout needs pidfiler and puf_constants in global env
    df = prep_tcout(calc)
    output_name = STACKDIR + reform_name + '_vs_' + base_name + '.parquet'
    df.to_parquet(output_name, engine='pyarrow')
    return None


# %% check: run 2017 law and 2018 law (x qbid limit) on default puf.csv
puf = pd.read_csv(LATEST_OFFICIAL_PUF)
recs_puf = tc.Records(data=puf)

# law2018xQlimit

clp = tc.Policy()
clp.implement_reform(eval('law2017'))
calc_clp = tc.Calculator(records=recs_puf, policy=clp)
calc_clp.advance_to_year(2018)
calc_clp.calc_all()
calc_clp.weighted_total('iitax') / 1e9  # 1719.791464209197

ref=tc.Policy()
ref.implement_reform(eval('law2018xQlimit'))
calc_ref = tc.Calculator(records=recs_puf, policy=ref)
calc_ref.advance_to_year(2018)
calc_ref.calc_all()
calc_ref.weighted_total('iitax') / 1e9  # 1521.8399218240877
# note: IRS number is ~1,538.749 table 1.1 Total income tax


calc_clp.weighted_total('iitax') / 1e9
calc_ref.weighted_total('iitax') / 1e9
(calc_clp.weighted_total('iitax') - calc_ref.weighted_total('iitax')) / 1e9  # 197.95154238510938


df = calc_clp.dataframe(variable_list=[], all_vars=True)
(df.iitax * df.s006).sum() / 1e9
(df.c00100 * df.s006).sum() / 1e9 #  11966


# %% get puf regrown reweighted data and create recs
puf2018 = pd.read_parquet(PUFDIR + 'puf2018_weighted' + '.parquet', engine='pyarrow')
puf2018.c00100.describe()

pidfiler = puf2018[['pid', 'filer']]

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


# %% quick checks on the puf
# 2018 IRS Table 1.2 filers:
    # number filers 153,774,296
    # agi  11643.439106 billion
pu.uvals(puf2018.columns)
# all returns
puf2018.s006.sum()  # 180,177,697
(puf2018.s006 * puf2018.c00100).sum() / 1e9  # 11895.191672684836
# equivalent all-records puf.csv value is  11966 $71b greater

# filers, 2017 definition
puf2018.query('filer == True').s006.sum()  # 154,586,946.9856476
(puf2018.query('filer == True').s006 * puf2018.query('filer == True').c00100).sum() / 1e9
# $11,807.2 billion agi -- too much -- but this is 2017 definition

temp = pd.merge(puf2018.loc[:, ['pid', 's006', 's006_default', 'c00100']],
                sweights_2018.loc[:, ['pid', 'weight']],
                on='pid', how='inner')
temp.s006.sum()  # 155345878
temp.weight.sum()
temp.s006_default.sum() # 147903590
(temp.weight * temp.c00100).sum() / 1e9  # 11,820

sweights_2018.weight.sum()



# %% reforms in isolation
# build the baseline
pol_base = tc.Policy()
pol_base.implement_reform(eval('law2017'))
calc_base = tc.Calculator(policy=pol_base, records=recs)
calc_base.calc_all()

# solo_reform(reform_name, base_name, calc_base)
STACKDIR = TCOUTDIR + 'unstacked/'  # define where to send the output - must be an existing dir
solo_reform('law2017', 'law2017', calc_base)
solo_reform('allrates2018', 'law2017', calc_base)
solo_reform('sd2018', 'law2017', calc_base)
solo_reform('persx2018', 'law2017', calc_base)
solo_reform('qbid2018_limitfalse', 'law2017', calc_base)
solo_reform('salt2018', 'law2017', calc_base)
solo_reform('amt2018', 'law2017', calc_base)
solo_reform('other_2018vs2017', 'law2017', calc_base)
solo_reform('law2018xQlimit', 'law2017', calc_base)

solo_reform('law2018', 'law2017', calc_base)


# %% stack reforms

# CAUTION: note that PT_qbid_limit_switch (set to False) will remain in effect
# from the time it is put into effect until the time it is turned off.
# Because it is NOT included in TCJA.json, which is the same as law2018,
# it will not be turned off by that reform. Thus, when we run the stacks of
# reforms below, when we get to law2018 PT_qbid_limit_switch will remain
# False and we will get different results for law2018 than we would if we
# ran law2018 in isolation.

# %% jct stacking order of reforms
order = 0
wsum_prior = 0
pol = tc.Policy()
STACKDIR = TCOUTDIR + 'stack_jct/'  # define where to send the output - must be an existing dir

add_reform('law2017')
add_reform('allrates2018')
add_reform('sd2018')
add_reform('persx2018')
add_reform('qbid2018_limitfalse')
add_reform('salt2018')
add_reform('amt2018')
add_reform('law2018')
order


# %% salt first stacking order of reforms
order = 0
wsum_prior = 0
pol = tc.Policy()
STACKDIR = TCOUTDIR + 'stack_saltfirst/'  # define where to send the output - must be an existing dir

add_reform('law2017')
add_reform('salt2018')
add_reform('sd2018')
add_reform('persx2018')
add_reform('qbid2018_limitfalse')
add_reform('amt2018')
add_reform('allrates2018')
add_reform('law2018')
order


# OLD orders before combining rates
# jct order
# add_reform('law2017')
# add_reform('rates2018')
# add_reform('sd2018')
# add_reform('persx2018')
# add_reform('passthrough_qbidxlimit2018')
# add_reform('salt2018')
# add_reform('amt2018')
# add_reform('law2018')

# salt first, rates last order
# add_reform('law2017')
# add_reform('salt2018')
# add_reform('sd2018')
# add_reform('persx2018')
# add_reform('passthrough_qbidxlimit2018')
# add_reform('amt2018')
# add_reform('rates2018')
# add_reform('law2018')



