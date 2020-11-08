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

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


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
    'c01000pos': 'cggross',
    'c01000neg': 'cgloss',
    'e01500': 'pensions',
    'e02400': 'socsectot',
    'c02500': 'socsectaxable',
    'e26270pos': 'partnerscorpinc',
    'e26270neg': 'partnerscorploss',

    # itemized deductions
    'c17000': 'id_medical_capped',
    'c18300': 'id_taxpaid',
    'c19200': 'id_intpaid',
    'c19700': 'id_contributions'
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
check  # 34 items
# np.setdiff1d(list(irspuf_fullmap.values()), check.pufvar.tolist())  # elements in first list NOT in second

target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
target_mappings.to_csv(DATADIR + 'target_mappings.csv', index=False)


# %% OLD BELOW HERE


# %% scratch area helper functions
def wsum(var, puf):
    val = (puf[var] * puf['s006']).sum()
    return val

def nret(var, puf):
    val = ((puf[var] != 0)* puf['s006']).sum()
    return val

def irsn(irsvar):
    irsvar = 'nret_' + irsvar
    q = 'common_stub==0 and variable==@irsvar'
    val = irstot.query(q)[['value']]
    return val.iat[0,0]

def irssum(irsvar):
    q = 'common_stub==0 and variable==@irsvar'
    val = irstot.query(q)[['value']]
    return val.iat[0,0]

# %% scratch marital status
irsvar = 'nret_single'  # 73,021,932
irsvar = 'nret_mfjss'  # 54,774,397
irsvar = 'nret_mfs'  # 3,212,807
irsvar = 'nret_hoh'  # 21,894,095
# print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

var = 'mars1'  # 75,537,981
var = 'mars2'  # 57,654,690
var = 'mars3'  # 2,370,429
var = 'mars4'  # 23,195,817
var = 'mars5'  # 0
print(f'{nret(var, puf[puf.filer]):,.0f}')
# print(f'{wsum(var, puf[puf.filer]):,.0f}')


# %% scratch taxable interest
# per pub 1304
# irs concept (line 8a, Form 1040), same as e00300
# we have way too many records with nonzero values but the weighted sum is
# in the right ballpark (4.6%)
# DON'T TARGET NUMBER OF NZ RETURNS

data = puf[puf.filer].e00300
weights = puf[puf.filer].s006

data = puf.query('filer and (e00300 != 0)')['e00300']
weights = puf.query('filer and (e00300 != 0)')['s006']

res = wp(data, weights, qtiles)
np.round(res, 1)
qtiles
type(res)

# from Anderson:
# Can you look where those units with a low value for e00300 are coming from
# the PUF or CPS? You can use the data_source variable to do it. It'll
# be 1 if the record is based on a PUF unit, 0 if it's from the CPS.
pufint = puf.query('filer and (e00300 != 0)')[['data_source', 's006', 'e00300']]
icuts =  [-9e99, 1.0, 100, 200, 500, 1000, 10e3, 100e3, 9e99]
pufint['intgroup'] = pd.cut(pufint['e00300'],
                           icuts,
                           labels=range(1, 9),
                           right=False)
intsums = pufint.groupby(['intgroup', 'data_source'])[['s006']].sum().reset_index()
intsums.info()
intsums.pivot(index='intgroup', columns=['data_source'], values='s006')

intsums = pufint.groupby(['intgroup', 'data_source'])[['s006']].sum().reset_index()
intsums.pivot(index='intgroup', columns=['data_source'], values='s006')

intsums = pufint[['intgroup', 'data_source', 'e00300']].groupby(['intgroup', 'data_source']).count().reset_index()
intsums.pivot(index='intgroup', columns=['data_source'], values='e00300')

capgains.loc[:, keepvars].pivot(index=idvars, columns=['variable'], values='value')

# %% scratch dividends
# irs ordinary dividends line 9a form 1040 same as e00600 this is the larger one so target this
# irs Qualified Dividends (line 9b, Form 1040) same as e00650
var = 'e00650'
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')


# %% scratch Social Security total income
irsvar = 'socsectot'  # 28,967,603 644,989,570,000
print(f'{irsn(irsvar):,.0f}'
print(f'{irssum(irsvar):,.0f}')

var = 'e02400'  # 38,568,626 841,444,755,391
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')

pufint = puf.query('filer and (e00300 != 0)')[['data_source', 's006', 'e00300']]
icuts =  [-9e99, 1.0, 100, 200, 500, 1000, 10e3, 100e3, 9e99]
pufint['intgroup'] = pd.cut(pufint['e00300'],
                           icuts,
                           labels=range(1, 9),
                           right=False)
intsums = pufint.groupby(['intgroup', 'data_source'])[['s006']].sum().reset_index()
intsums.info()
intsums.pivot(index='intgroup', columns=['data_source'], values='s006')

intsums = pufint.groupby(['intgroup', 'data_source'])[['s006']].sum().reset_index()
intsums.pivot(index='intgroup', columns=['data_source'], values='s006')

intsums = pufint[['intgroup', 'data_source', 'e00300']].groupby(['intgroup', 'data_source']).count().reset_index()
intsums.pivot(index='intgroup', columns=['data_source'], values='e00300')


# %% scratch partnership and S corp
# e02000 Sch E total rental, royalty, partnership, S-corporation, etc, income/loss (includes e26270 and e27200)
# e26270 Sch E: Combined partnership and S-corporation net income/loss (includes k1bx14p and k1bx14s amounts and is included in e02000)

# nret_partnerscorpinc Number of returns with partnership or S corporation net income
# partnerscorpinc Partnership or S corporation net income
# nret_partnerscorploss Number of returns with partnership or S corporation net loss
# partnerscorploss Partnership or S corporation net loss

irsvar = 'partnerscorpinc'  # 6,240,408, 833,430,151,000
irsvar = 'partnerscorploss'  # 2,872,745, 153,150,406,000
print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

var = 'e26270pos'  # 5,832,109  681,710,267,190
var = 'e26270neg'  # 3,138,926  -164,846,548,501
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')


# %% scratch medical deductions
# id_medical_capped
# e17500 Description: Itemizable medical and dental expenses. WARNING: this variable is zero below the floor in PUF data.
# c17000 Sch A: Medical expenses deducted (component of pre-limitation c21060 total)
# I don't understand why c17000 is called component of pre-limitation -- it does appear to be limited
# e17500_capped Sch A: Medical expenses, capped as a decimal fraction of AGI
# irs 17in21id.xls pre-limit total 155,408,904  10,171,257
# id_medical_capped irs 17in21id.xls limited 102,533,387  10,171,257

irsvar = 'id_medical_capped'
print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

var = 'e17500' # 200,511,523,398  17,563,931
var = 'c17000'  # 96,675,292,760  9,725,100
var = 'e17500_capped'  # 200,511,523,398 17,563,931
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')

# seems like I should match c17000 against id_medical_capped the limited deduction in the irs data


# %% scratch SALT
var = 'c18300'  # 45,959,926 585,237,496,064
var = 'e18400'  # 153,487,189 513,295,087,752
var = 'e18400_capped' # 153,487,189 513,295,087,752
var = 'e18500_capped'
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')


# SALT
# irs values  nrets
# irs 17in21id.xls taxes paid deduction 624,820,806  46,431,232
# irs 17in21id.xls income tax  368,654,631
# irs 17in21id.xls sales tax   20,734,779
# irs 17in21id.xls real estate 222,237,629
# irs 17in21id.xls personal property taxes 10,679,233
# irs 17in21id.xls other taxes 2,514,534

# puf values (2017)
# note: c21060 is Itemized deductions before phase-out (zero for non-itemizers)

# c18300 Sch A: State and local taxes plus real estate taxes deducted (component of pre-limitation c21060 total)

# e18400 Itemizable state and local income/sales taxes

# e18400_capped Sch A: State and local income taxes deductible, capped as a decimal fraction of AGI
# 526,195,784,967

# e18500 Itemizable real-estate taxes paid  285,719,544,931
# e18500_capped Sch A: State and local real estate taxes deductible, capped as a decimal fraction of AGI
# 285,719,544,931


# %% scratch interest paid deduction
irsvar = 'id_mortgage' # 292,557,787,000 33,746,351
irsvar = 'id_intpaid' #  313,944,112,000  34,327,403
print(f'{irsn(irsvar):,.0f}')
print(f'{irssum(irsvar):,.0f}')

# e19200 Description: Itemizable interest paid
# c19200 Sch A: Interest deducted (component of pre-limitation c21060 total)
# e19200_capped Sch A: Interest deduction deductible, capped as a decimal fraction of AGI
var = 'e19200' #  424,406,109,267  55,333,072
var = 'c19200' # 357,486,840,616  36,146,781
var = 'e19200_capped'  # 424,406,109,267 55,333,072
print(f'{wsum(var):,.0f}')
print(f'{nret(var):,.0f}')
# seems like I should link c19200 to id_intpaid


# %% scratch charitable contributions

irsvar = 'id_contributions'
print(f'{irsn(irsvar):,.0f}')  # 37,979,015
print(f'{irssum(irsvar):,.0f}') # 256,064,685,000

# e20100 Itemizable charitable giving: other than cash/check contributions. WARNING: this variable is already capped in PUF data.

# c19700 Sch A: Charity contributions deducted (component of pre-limitation c21060 total)
# e19800_capped Sch A: Charity cash contributions deductible, capped as a decimal fraction of AGI
# e19800 Itemizable charitable giving: cash/check contributions. WARNING: this variable is already capped in PUF data.
# e20100_capped Sch A: Charity noncash contributions deductible, capped as a decimal fraction of AGI
var = 'e19800'  # 212,635,455,351   101,903,175  # cash
var = 'e20100'  # 64,207,135,577   56,359,659  # noncash
var = 'c19700' # 38,553,297 211,073,849,881
var = 'e20100_capped' # 64,207,135,577  56,359,659  # capped
print(f'{nret(var, puf[puf.filer]):,.0f}')
print(f'{wsum(var, puf[puf.filer]):,.0f}')

# growfactor for e19800 and e20100 is ATXPY
# seems like we could use the sum of these e19800, e20100 as roughly equiv of id_contributions?
# for now match c19700 to id_contributions ??

# post issue on github



