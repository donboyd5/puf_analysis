# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 04:24:19 2020

@author: donbo
"""

# %% target notes
# puf counts generally are my definition of filers
# in addition to the obvious items


# filing status - just target the 3 biggest statuses
# they look close enough to work
# nret_all nret_all
# nret_single mars1
# nret_mfjss mars2
# nret_hoh mars4

# income items

# agi, c00100, target both nnz and amount
# wages, e00200, target both nnz and amount

# taxable interest income
# taxint irs line 8a, Form 1040, same as e00300
# DON'T TARGET NUMBER OF NZ RETURNS

# ordinary dividends line 9a form 1040
# orddiv same as e00600 the larger one
# looks like I should reduce the growfactor
# then should be ok to target both nnz and amount

# capital gains net income
# cgnet and c01000
# should be good to target both nnz and amount

# pensions total income
# pensions, e01500
# amount good, too many nnz in puf but PROB should target both

# social security taxable income
# socsectaxable, c02500
# looks like I should target both nnz and amount
# target taxable, not total

# partnership and scorp income and loss
# partnerscorpinc e26270pos
# partnerscorploss e26270neg
# should be able to target both nnz and amount for both

# itemized deductions
# irs id_medical_capped, c17000 the limited deduction in the irs data
# target both nnz and amount

# taxes paid deduction
# id_taxpaid, c18300  target both nnz and amount
# irs 17in21id.xls taxes paid deduction  46,431,232 624,820,806
# c18300 Sch A: State and local taxes plus real estate taxes deducted
#   (component of pre-limitation c21060 total)
# c18300  45,959,926 585,237,496,064
# wouldn't hurt to increase the growfactor

# interest paid deduction
# id_intpaid, c19200  should be able to target both nnz and amount

# charitable contributions deduction
# id_contributions c19700 appears correct

# item deds only for those with c04470 > 0 (i.e., itemizers)
# c00100, e00200, e00300, e00600, c01000, e01500, c02500, e26270pos, e26270neg, c17000, c18300, c19200, c19700


# %% imports
import pandas as pd
import numpy as np
import puf_constants as pc
from datetime import date
import json

import puf_utilities as pu


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'
TCOUTDIR = PUFDIR + 'taxcalc_output/'

PUF_DEFAULT = TCOUTDIR + 'puf2017_default.parquet'
PUF_REGROWN = TCOUTDIR + 'puf2017_regrown.parquet'


# %% functions
def get_nnz(puf, pufvars_to_nnz):
    for var in pufvars_to_nnz:
        puf[var + '_nnz'] = puf[var].ne(0) * 1
    return puf


# %% ONETIME create and save puf-variables and irs-variables linkages
# pufirs_fullmap and irspuf_fullmap are the important outputs of this section
# map ALL variables we think we can map
# we can choose a subset for comparison or for reweighting
# this mapping will help us define the nnz and sum variables we need to create
# in the puf
# and the related full mappings to create
pufirs_map = {
    # number of returns and marital status indicators
    'nret_all': 'nret_all',
    'mars1': 'nret_single',
    'mars2': 'nret_mfjss',
    'mars3': 'nret_mfs',
    'mars4': 'nret_hoh',

    # agi income components
    'c00100': 'agi',
    'e00200': 'wages',
    'e00300': 'taxint',
    'e00600': 'orddiv',
    'c01000': 'cgnet',  # we must create IRS variable cgnet from cggross and cgloss
    'c01000pos': 'cggross',
    'c01000neg': 'cgloss',
    'e01500': 'pensions',
    'e02400': 'socsectot',
    'c02500': 'socsectaxable',

    # business-like income and losses
    'e00900': 'busprofnet',
    'e00900pos': 'busprofnetinc',
    'e00900neg': 'busprofnetloss',
    # e02000 Rents, royalties, parternship, S corp, estates/trusts, etc.
    'e02000': 'e02000',
    'e02000pos': 'e02000inc',
    'e02000neg': 'e02000loss',
    # CAUTION: partnerscorpinc is included in e02000
    'e26270pos': 'partnerscorpinc',
    'e26270neg': 'partnerscorploss',

    # itemized deductions
    'c17000': 'id_medical_capped',
    'c18300': 'id_taxpaid',
    'c19200': 'id_intpaid',
    'c19700': 'id_contributions',

    # taxable income concepts
    'c04800': 'ti',  # Taxable income, regular

    # tax concepts
    'c05800': 'taxbc',  # Income tax before credits (IRS definition, not PUF definition)
    # Income tax after credits, max(0, c09200 - niit - refund) for now
    # after tax-calculator is patched, we'll use max(0, c09200 - refund)
    'taxac_irs': 'taxac',
    # other tax variables do not map directly and would have to be constructed
    }
# CAUTION: reverse xwalk relies on having only one keyword per value
irspuf_map = {val: kw for kw, val in pufirs_map.items()}

# define the variables we will transform by creating nnz versions
puf_vars = pufirs_map.keys()
irs_vars = pufirs_map.values()
# puf_default[puf_vars]

pufvars_to_nnz = [s for s in puf_vars if ('nret' not in s) and ('mars' not in s) and ('c00100' != s)]
irsvars_to_nret = [s for s in irs_vars if ('nret' not in s) and ('agi' != s)]

nnz_vars =  [s + '_nnz' for s in pufvars_to_nnz]
nret_vars =  ['nret_' + s for s in irsvars_to_nret]

# make a map in the desired order
plist = []
ilist = []
for p, i in zip(puf_vars, irs_vars):
    plist.append(p)
    ilist.append(i)
    if p in pufvars_to_nnz:
        plist.append(p + '_nnz')
        ilist.append('nret_' + i)

len(plist) == len(ilist)
pufirs_fullmap = {plist[i]: ilist[i] for i in range(len(plist))}

# CAUTION: reverse xwalk relies on having only one keyword per value
irspuf_fullmap = {val: kw for kw, val in pufirs_fullmap.items()}

# save dicts
json.dump(pufirs_fullmap, open(DATADIR + 'pufirs_fullmap.json', 'w'))
check = json.load(open(DATADIR + 'pufirs_fullmap.json'))


# %% ONETIME: define and save possible targets
IRSDAT = DATADIR + 'targets2017_collapsed.csv'
irstot = pd.read_csv(IRSDAT)
irstot
irstot.info()
irstot.count()
irsvars = irstot.variable.value_counts().sort_index()

# put the puf variable name on irstot
targets_possible = irstot.copy()
targets_possible['pufvar'] = targets_possible.variable.map(irspuf_fullmap)
targets_possible = targets_possible.dropna(subset=['pufvar'])
targets_possible.rename(columns={'variable': 'irsvar', 'value': 'irs'}, inplace=True)
# change column order
cols = ['common_stub', 'incrange', 'pufvar', 'irsvar', 'irs', 'table_description', 'column_description', 'src', 'excel_column']
targets_possible = targets_possible[cols]
# sort in the order of the mapping
targets_possible['pufvar'] = pd.Categorical(targets_possible['pufvar'],
                                            categories=pufirs_fullmap.keys(),
                                            ordered=True)
targets_possible = targets_possible.sort_values(by=['common_stub', 'pufvar'], axis=0)

targets_possible.to_csv(DATADIR + 'targets2017_possible.csv', index=False)

check = targets_possible[['irsvar', 'pufvar']].drop_duplicates()
check  # 50 items
# np.setdiff1d(list(irspuf_fullmap.values()), check.pufvar.tolist())  # elements in first list NOT in second

target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
target_mappings.to_csv(DATADIR + 'target_mappings.csv', index=False)

