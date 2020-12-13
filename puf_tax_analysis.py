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

instructions from my conversation with Matt 11/28/2020 about installing
Tax-Calculator from sournce on my own machine

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


# %% define: reforms versus 2017 law going forward toward 2018 law -- file names or dicts
# Note: # tc.Policy().read_json_reform(qbid_limit) -- creates a dict out of
# a json file, converting false or "false" to False -- so I can create dicts directly
salt2018 = {"ID_AllTaxes_c": {"2018": [10000.0, 10000.0, 5000.0, 10000.0, 10000.0]}}

sd2018 = {"STD": {"2018": [12000, 24000, 12000, 18000, 24000]}}

persx2018 = {"II_em": {"2018": 0}}

# combine all rates
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

amt2018 = {"AMT_em": {"2018": [70300, 109400, 54700, 70300, 109400]},
           "AMT_em_ps": {"2018": [500000, 1000000, 500000, 500000, 1000000]},
           "AMT_em_pe": {"2018": 718800}}

# caution: this next reform is based on carefully deleting provisions already estimated
# above. needs to be checked/reviewed
other_2018vs2017 = tc.Policy.read_json_reform(REFDIR + 'other_2018vs2017.json')


# %% define: reforms versus 2018 law, going back toward 2017 law -- file names or dicts
# run against 2018 law, these take away a reform

salt2017 = {"ID_AllTaxes_c": {"2018": [9e99, 9e99, 9e99, 9e99, 9e99]}}

# std deduction - will it be sufficient to use 2017 and let tc move it to 2018? I think so
# that was the way it works against 2017 law when adding it
sd2017 = {"STD": {"2017": [6350, 12700, 6350, 9350, 12700]}} # use 2017 and index to 2018
# or the following?? the former was all that was in TCJA.json
# sd2017 = {"STD": {"2017": [6350, 12700, 6350, 9350, 12700]},
#           "STD_Dep": {"2017": 1050},
#           "STD_Aged": {"2017": [1550, 1250, 1250, 1550, 1550]}}

persx2017 = {"II_em": {"2017": 4050}}  # use 2017 and index to 2018
# what about:
    #     "II_em_ps": {"2017": [261500, 313800, 156900, 287650, 313800]},

# set amt values at 2017 levels so they will be indexed
amt2017 = {"AMT_em": {"2017": [54300.0, 84500.0, 42250.0, 54300.0, 84500.0]},
           "AMT_em_ps": {"2017": [120700.0, 160900.0, 80450.0, 120700.0, 160900.0]},
           "AMT_em_pe": {"2017": 249450.0}}


# %% define full reforms -- file names or dicts
law2017 = tc.Policy.read_json_reform(REFDIR + '2017_law.json')
law2018 = tc.Policy.read_json_reform(REFDIR + 'TCJA.json')
law2018xQlimit = {**law2018, **qbid_limitfalse}  # TCJA but with qbid limit set to false


# %% selected parameter descriptions

# AMT_em
# Description: The amount of AMT taxable income exempted from AMT.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# for: [single, mjoint, mseparate, headhh, widow]
# 2013: [51900.0, 80800.0, 40400.0, 51900.0, 80800.0]
# 2014: [52800.0, 82100.0, 41050.0, 52800.0, 82100.0]
# 2015: [53600.0, 83400.0, 41700.0, 53600.0, 83400.0]
# 2016: [53900.0, 83800.0, 41900.0, 53900.0, 83800.0]
# 2017: [54300.0, 84500.0, 42250.0, 54300.0, 84500.0]
# 2018: [70300.0, 109400.0, 54700.0, 70300.0, 109400.0]
# 2019: [71700.0, 111700.0, 55850.0, 71700.0, 111700.0]
# 2020: [72804.18, 113420.18, 56710.09, 72804.18, 113420.18]
# 2021: [73335.65, 114248.15, 57124.07, 73335.65, 114248.15]
# 2022: [74032.34, 115333.51, 57666.75, 74032.34, 115333.51]
# 2023: [75202.05, 117155.78, 58577.88, 75202.05, 117155.78]
# 2024: [76653.45, 119416.89, 59708.43, 76653.45, 119416.89]
# 2025: [78186.52, 121805.23, 60902.6, 78186.52, 121805.23]
# 2026: [62898.0, 97880.0, 48940.0, 62898.0, 97880.0]
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error

# AMT_em_pe¶
# Description: The AMT exemption is entirely disallowed beyond this AMT taxable income level for individuals who are married but filing separately.
# Has An Effect When Using: PUF data: True CPS data: False
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# 2013: 238550.0
# 2014: 242450.0
# 2015: 246250.0
# 2016: 247450.0
# 2017: 249450.0
# 2018: 718800.0
# 2019: 733700.0
# 2020: 744998.98
# 2021: 750437.47
# 2022: 757566.63
# 2023: 769536.18
# 2024: 784388.23
# 2025: 800075.99
# 2026: 288949.0
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error

# AMT_prt
# Description: AMT exemption will decrease at this rate for each dollar of AMT taxable income exceeding AMT phaseout start.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: False Is Inflation Indexed: False
# Value Type: float
# Known Values:
# 2013: 0.25
# 2014: 0.25
# 2015: 0.25
# 2016: 0.25
# 2017: 0.25
# 2018: 0.25
# 2019: 0.25
# Valid Range: min = 0 and max = 1
# Out-of-Range Action: error

# AMT_em_ps
# Description: AMT exemption starts to decrease when AMT taxable income goes beyond this threshold.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# for: [single, mjoint, mseparate, headhh, widow]
# 2013: [115400.0, 153900.0, 76950.0, 115400.0, 153900.0]
# 2014: [117300.0, 156500.0, 78250.0, 117300.0, 156500.0]
# 2015: [119200.0, 158900.0, 79450.0, 119200.0, 158900.0]
# 2016: [119700.0, 159700.0, 79850.0, 119700.0, 159700.0]
# 2017: [120700.0, 160900.0, 80450.0, 120700.0, 160900.0]
# 2018: [500000.0, 1000000.0, 500000.0, 500000.0, 1000000.0]
# 2019: [510300.0, 1020600.0, 510300.0, 510300.0, 1020600.0]
# 2020: [518158.62, 1036317.24, 518158.62, 518158.62, 1036317.24]
# 2021: [521941.18, 1043882.36, 521941.18, 521941.18, 1043882.36]
# 2022: [526899.62, 1053799.24, 526899.62, 526899.62, 1053799.24]
# 2023: [535224.63, 1070449.27, 535224.63, 535224.63, 1070449.27]
# 2024: [545554.47, 1091108.94, 545554.47, 545554.47, 1091108.94]
# 2025: [556465.56, 1112931.12, 556465.56, 556465.56, 1112931.12]
# 2026: [139812.0, 186378.0, 93189.0, 139812.0, 186378.0]
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error



# II_em
# Description: Subtracted from AGI in the calculation of taxable income, per taxpayer and dependent.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# 2013: 3900.0
# 2014: 3950.0
# 2015: 4000.0
# 2016: 4050.0
# 2017: 4050.0
# 2018: 0.0
# 2019: 0.0
# 2020: 0.0
# 2021: 0.0
# 2022: 0.0
# 2023: 0.0
# 2024: 0.0
# 2025: 0.0
# 2026: 4691.0
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error

# II_prt
# Description: Personal exemption amount will decrease by this rate for each dollar of AGI exceeding exemption phaseout start.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: False Is Inflation Indexed: False
# Value Type: float
# Known Values:
# 2013: 0.02
# 2014: 0.02
# 2015: 0.02
# 2016: 0.02
# 2017: 0.02
# 2018: 0.02
# 2019: 0.02
# Valid Range: min = 0 and max = 1
# Out-of-Range Action: error

# II_em_ps
# Description: If taxpayers’ AGI is above this level, their personal exemption will start to decrease at the personal exemption phaseout rate (PEP provision).
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# for: [single, mjoint, mseparate, headhh, widow]
# 2013: [250000.0, 300000.0, 150000.0, 275000.0, 300000.0]
# 2014: [254200.0, 305050.0, 152525.0, 279650.0, 305050.0]
# 2015: [258250.0, 309900.0, 154950.0, 284040.0, 309900.0]
# 2016: [259400.0, 311300.0, 155650.0, 285350.0, 311300.0]
# 2017: [261500.0, 313800.0, 156900.0, 287650.0, 313800.0]
# 2018: [9e+99, 9e+99, 9e+99, 9e+99, 9e+99]
# 2019: [9e+99, 9e+99, 9e+99, 9e+99, 9e+99]



# STD
# Description: Amount filing unit can use as a standard deduction.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# for: [single, mjoint, mseparate, headhh, widow]
# 2013: [6100.0, 12200.0, 6100.0, 8950.0, 12200.0]
# 2014: [6200.0, 12400.0, 6200.0, 9100.0, 12400.0]
# 2015: [6300.0, 12600.0, 6300.0, 9250.0, 12600.0]
# 2016: [6300.0, 12600.0, 6300.0, 9300.0, 12600.0]
# 2017: [6350.0, 12700.0, 6350.0, 9350.0, 12700.0]
# 2018: [12000.0, 24000.0, 12000.0, 18000.0, 24000.0]
# 2019: [12200.0, 24400.0, 12200.0, 18350.0, 24400.0]
# 2020: [12387.88, 24775.76, 12387.88, 18632.59, 24775.76]
# 2021: [12478.31, 24956.62, 12478.31, 18768.61, 24956.62]
# 2022: [12596.85, 25193.71, 12596.85, 18946.91, 25193.71]
# 2023: [12795.88, 25591.77, 12795.88, 19246.27, 25591.77]
# 2024: [13042.84, 26085.69, 13042.84, 19617.72, 26085.69]
# 2025: [13303.7, 26607.4, 13303.7, 20010.07, 26607.4]
# 2026: [7355.0, 14711.0, 7355.0, 10831.0, 14711.0]
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error

# STD_Aged
# Description: To get the standard deduction for aged or blind individuals, taxpayers need to add this value to regular standard deduction.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# for: [single, mjoint, mseparate, headhh, widow]
# 2013: [1500.0, 1200.0, 1200.0, 1500.0, 1500.0]
# 2014: [1550.0, 1200.0, 1200.0, 1550.0, 1550.0]
# 2015: [1550.0, 1250.0, 1250.0, 1550.0, 1550.0]
# 2016: [1550.0, 1250.0, 1250.0, 1550.0, 1550.0]
# 2017: [1550.0, 1250.0, 1250.0, 1550.0, 1550.0]
# 2018: [1600.0, 1300.0, 1300.0, 1600.0, 1300.0]
# 2019: [1650.0, 1300.0, 1300.0, 1650.0, 1300.0]
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error

# STD_Dep
# Description: This is the maximum standard deduction for dependents.
# Has An Effect When Using: PUF data: True CPS data: True
# Can Be Inflation Indexed: True Is Inflation Indexed: True
# Value Type: float
# Known Values:
# 2013: 1000.0
# 2014: 1000.0
# 2015: 1050.0
# 2016: 1050.0
# 2017: 1050.0
# 2018: 1050.0
# 2019: 1100.0
# Valid Range: min = 0 and max = 9e+99
# Out-of-Range Action: error





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

# 2018 law, setting QBID limit to False
ref=tc.Policy()
ref.implement_reform(eval('law2018xQlimit'))
calc_ref = tc.Calculator(records=recs_puf, policy=ref)
calc_ref.advance_to_year(2018)
calc_ref.calc_all()
calc_ref.weighted_total('iitax') / 1e9  # 1521.8399218240877
# note: IRS number is ~1,538.749 table 1.1 Total income tax

# 2018 law default - WITHOUT setting QBID limit to False
ref2=tc.Policy()
ref2.implement_reform(eval('law2018'))
calc_ref2 = tc.Calculator(records=recs_puf, policy=ref2)
calc_ref2.advance_to_year(2018)
calc_ref2.calc_all()
calc_ref2.weighted_total('iitax') / 1e9  # 1557.3596890387046


calc_clp.weighted_total('iitax') / 1e9
calc_ref.weighted_total('iitax') / 1e9
(calc_clp.weighted_total('iitax') - calc_ref.weighted_total('iitax')) / 1e9  # 197.95154238510938


df = calc_clp.dataframe(variable_list=[], all_vars=True)
df['filer'] = pu.filers(df, year=2018)
(df.iitax * df.s006).sum() / 1e9
(df.c00100 * df.s006).sum() / 1e9 #  11966

df.query('filer == True').s006.sum()  # 147,804,768 not 154,586,946.9856476
(df.query('filer == True').s006 * df.query('filer == True').c00100).sum() / 1e9
# 11890.573620761066


# %% calc: get puf regrown reweighted data and create recs
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


# %% check: quick checks on puf2018 regrown reweighted
# 2018 IRS Table 1.2 filers:
    # number filers 153,774,296
    # agi  11643.439106 billion
pu.uvals(puf2018.columns)
# all returns
puf2018.s006.sum()  # 180,177,697
(puf2018.s006 * puf2018.c00100).sum() / 1e9  # 11895.191672684836
# equivalent all-records puf.csv value is  11966 $71b greater

# filers
puf2018.query('filer == True').s006.sum()  # 154,586,946.9856476
(puf2018.query('filer == True').s006 * puf2018.query('filer == True').c00100).sum() / 1e9
# $11,807.2 billion agi -- too much -- ?? but this is 2017 definition
# $11,890 is puf value

temp = pd.merge(puf2018.loc[:, ['pid', 's006', 's006_default', 'c00100']],
                sweights_2018.loc[:, ['pid', 'weight']],
                on='pid', how='inner')
temp.s006.sum()  # 155345878
temp.weight.sum()
temp.s006_default.sum() # 147903590
(temp.weight * temp.c00100).sum() / 1e9  # 11,820

sweights_2018.weight.sum()


# %% calc: records object
recs = tc.Records(data=puf2018,
                  start_year=2018,
                  weights=weights_us,
                  adjust_ratios=None)
 # note that we don't need to advance because start year is 2018


# %% calc: reforms versus 2017 law in isolation
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

# for informational purposes only:
solo_reform('law2018', 'law2017', calc_base)  # law2018 has qbid switch True - it is TCJA.json


# %% calc: reforms versus 2018 law in isolation
# build the baseline
pol_base = tc.Policy()
pol_base.implement_reform(eval('law2018xQlimit'))  # full 2018 law with qbid limit false
calc_base = tc.Calculator(policy=pol_base, records=recs)
calc_base.calc_all()
calc_base.weighted_total('iitax') / 1e9  # 1466.4086894196798 -- good

# solo_reform(reform_name, base_name, calc_base)
STACKDIR = TCOUTDIR + 'solo_vs_2018law/'  # define where to send the output - must be an existing dir
solo_reform('law2017', 'law2018xQlimit', calc_base)  # a full trip backward
solo_reform('sd2017', 'law2018xQlimit', calc_base)
solo_reform('salt2017', 'law2018xQlimit', calc_base)
solo_reform('persx2017', 'law2018xQlimit', calc_base)
solo_reform('amt2017', 'law2018xQlimit', calc_base)
solo_reform('law2018xQlimit', 'law2018xQlimit', calc_base)  # trip to nowhere - just 2018 law, with qbid switch false


# %% check: total tax comparison
# temp = pd.read_parquet(TCOUTDIR + 'unstacked/law2018xQlimit_vs_law2017.parquet', engine='pyarrow')
# temp['taxcomp'] = np.where((temp.c09200 - temp.refund) < 0, 0, temp.c09200 - temp.refund)

# irstottax = 1538749447 / 1e6
# iitax = (temp.iitax * temp.s006).sum() / 1e9
# taxcomp = (temp.taxcomp * temp.s006).sum() / 1e9
# taxcomp
# iitax
# irstottax
# iitax / irstottax * 100 - 100
# taxcomp / irstottax * 100 - 100


# %% info: stack reforms

# CAUTION: note that PT_qbid_limit_switch (set to False) will remain in effect
# from the time it is put into effect until the time it is turned off.
# Because it is NOT included in TCJA.json, which is the same as law2018,
# it will not be turned off by that reform. Thus, when we run the stacks of
# reforms below, when we get to law2018 PT_qbid_limit_switch will remain
# False and we will get different results for law2018 than we would if we
# ran law2018 in isolation.

# %% calc: jct stacking order of reforms
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


# %% calc: salt first stacking order of reforms
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


