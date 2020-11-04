# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 04:19:27 2020

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


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
HDFDIR = r'C:\programs_python\puf_analysis\ignore/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'

BASE_NAME = 'puf_adjusted'
BASE_NAME = 'puf2017_regrown_2020-11-02'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast


# %% constants
# pc.HT2_AGI_STUBS
# pc.ht2stubs
# pc.IRS_AGI_STUBS
# pc.irsstubs

qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% get target data
IRSDAT = DATADIR + 'targets2017_collapsed.csv'
irstot = pd.read_csv(IRSDAT)
irstot
irstot.info()
irstot.count()
irsvars = irstot.variable.value_counts().sort_index()


# %% create cgnet and nret_cgnet
# cgnet = cggross - cgloss, we will match against c01000
# we will calculate nret_cgnet ASSUMING it is nret_cggross + nret_cgloss
cgvars = ['cggross', 'cgloss', 'nret_cggross', 'nret_cgloss']
capgains = irstot.query('variable in @cgvars')
idvars = ['src', 'common_stub', 'incrange', 'table_description']
keepvars = idvars + ['variable', 'value']
cgwide = capgains.loc[:, keepvars].pivot(index=idvars, columns=['variable'], values='value')
cgwide['cgnet'] = cgwide.cggross - cgwide.cgloss
cgwide['nret_cgnet'] = cgwide.nret_cggross + cgwide.nret_cgloss
# cgwide = cgwide.rename(columns={'nret_cggross': 'nret_cgnet'}).reset_index()
cgwide = cgwide.reset_index()
cgwide = cgwide.drop(columns=['cggross', 'cgloss'])
cglong = cgwide.melt(id_vars=idvars)

# set column_description
# faster approach
ret_lab = 'Number of returns with capital gains net taxable ASSUMED = nret_cggross + nret_cgloss'
val_lab = 'Capital gains net taxable CALCULATED as gross - loss'
cglong.loc[cglong['variable'] == 'nret_cgnet', 'column_description'] = ret_lab
cglong.loc[cglong['variable'] == 'cgnet', 'column_description'] =val_lab

# alternative easy to understand but slow approach
# def f(row):
#     # not vectorized, only good for small data frames
#     if row['variable'] == 'cgnet':
#         label = 'Capital gains net taxable CALCULATED as gross - loss'
#     elif row['variable'] == 'nret_cgnet':
#         label = 'Number of returns with capital gains net taxable ASSUMED = nret_cggross'
#     return label
# cglong['column_description'] = cglong.apply(f, axis=1)


# %% update irstot
irstot = irstot.append(cglong)


# %% get the puf
# puf = pd.read_hdf(IGNOREDIR + 'puf2017_2020-10-26.h5')  # 1 sec
puf = pd.read_hdf(IGNOREDIR + 'puf2017_2020-11-02.h5')  # 1 sec

puf['common_stub'] = pd.cut(
    puf['c00100'],
    pc.COMMON_STUBS,
    labels=range(1, 19),
    right=False)
puf.info()

puf.head()['c00100']
puf.head()['e00600']


# %% create needed puf variables
puf['nret_all'] = 1

# marital status indicators
puf['mars1'] = puf.MARS.eq(1)
puf['mars2'] = puf.MARS.eq(2)
puf['mars3'] = puf.MARS.eq(3)
puf['mars4'] = puf.MARS.eq(4)
puf['mars5'] = puf.MARS.eq(5)

# partnership and S corp e26270
puf['e26270pos'] = puf.e26270 * puf.e26270.gt(0)
puf['e26270neg'] = puf.e26270 * puf.e26270.lt(0)


# %% define filers
# age_head, age_spouse
# np.quantile(puf.age_head, qtiles)  # 1 to 85 (1??)
# np.quantile(puf.age_spouse, qtiles)  # 50 to 97
# puf.MARS.value_counts()

puf['filer'] = filers(puf, 2017)
puf.filer.sum()
pufvars = puf.columns.sort_values()


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


# %% get nz counts and weighted sums of most puf variables, for FILERS
# get the subset of variables we want
# puf['common_stub']
# c18300 appears to be the SALT concept that corresponds to the uncapped deduction and comes a little
# close to what is in the irs spreadsheet
# c18300 Sch A: State and local taxes plus real estate taxes deducted (component of pre-limitation c21060 total)

keepcols = ('pid', 'common_stub', 's006', 'c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e01500', 'e02400', 'c02500',
            # itemized deductions
            'c17000', 'c18300', 'c19200', 'c19700')

pufsub = puf.loc[puf.filer, keepcols]

# make a long file with weighted values
puflong = pufsub.melt(id_vars=('pid', 'common_stub', 's006'))
puflong['nnz'] = (puflong.value != 0) * puflong.s006
puflong['wsum'] = puflong.value * puflong.s006

# get the sums by income range, add grand sums, add stub names
pufsums = puflong.groupby(['common_stub', 'variable'])[['nnz', 'wsum']].sum().reset_index()
grand_sums = pufsums.groupby(['variable']).sum().reset_index()
grand_sums['common_stub'] = 0
pufsums = pufsums.append(grand_sums)
pufsums = pd.merge(pufsums, pc.irsstubs, on=['common_stub'])
pufsums = pufsums.sort_values(['variable', 'common_stub'])
# reorder vars
vars = ['common_stub', 'incrange', 'variable', 'nnz', 'wsum']
pufsums = pufsums[vars]


# %% make long pufsums and map pufnames to irstot names
pufsumslong = pufsums.melt(id_vars=('common_stub', 'incrange', 'variable'), var_name='measure')
pufsumslong['puf_varmeas'] = pufsumslong.variable + '_' + pufsumslong.measure
pufsumslong.puf_varmeas.value_counts()

vmap = {# agi income components
        'c00100_nnz': 'nret_all',
        'c00100_wsum': 'agi',
        'e00200_nnz': 'nret_wages',
        'e00200_wsum': 'wages',
        'e00300_nnz': 'nret_taxint',
        'e00300_wsum': 'taxint',
        'e00600_nnz': 'nret_orddiv',
        'e00600_wsum': 'orddiv',
        'c01000_nnz': 'nret_cgnet',
        'c01000_wsum': 'cgnet',
        'e01500_nnz': 'nret_pensions',
        'e01500_wsum': 'pensions',
        'e02400_nnz': 'nret_socsectot',
        'e02400_wsum': 'socsectot',
        'c02500_nnz': 'nret_socsectaxable',
        'c02500_wsum': 'socsectaxable',
        # itemized deductions
        'c17000_nnz': 'nret_id_medical_capped',
        'c17000_wsum': 'id_medical_capped',
        'c18300_nnz': 'nret_id_taxpaid',
        'c18300_wsum': 'id_taxpaid',
        'c19200_nnz': 'nret_id_intpaid',
        'c19200_wsum': 'id_intpaid',
        'c19700_nnz': 'nret_id_contributions',
        'c19700_wsum': 'id_contributions'
        }

pufsumslong['irsvar'] = pufsumslong.puf_varmeas.map(vmap)
pufsumslong


# %% merge targets and pufsums, calc differences
irstot.info()
pufsumslong.info()
pufsumslong.puf_varmeas.value_counts()
comp = pd.merge(irstot.rename(columns={'variable': 'irsvar', 'value': 'irs'}),
                pufsumslong.rename(columns={'variable': 'pufvar', 'value': 'puf'}),
                on=['common_stub', 'incrange', 'irsvar'])
comp['diff'] = comp['puf'] - comp['irs']
comp['pdiff'] = comp['diff'] / comp['irs'] * 100
# reorder
mainvars = ['common_stub', 'incrange', 'irsvar', 'puf_varmeas',
            'irs', 'puf', 'diff', 'pdiff']
infovars = ['column_description', 'table_description', 'src']
comp = comp[mainvars + infovars]
comp.info()

comp.puf_varmeas.value_counts()


# %% print or write results

s = comp.copy()[mainvars + infovars]
# define custom sort order
s['puf_varmeas'] = pd.Categorical(s['puf_varmeas'], categories=vmap.keys())
s = s.sort_values(by=['puf_varmeas', 'common_stub'])

s['pdiff'] = s['pdiff'] / 100.0
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:.1%}'}
for key, value in format_mapping.items():
    s[key] = s[key].apply(value.format)

vlist = s.puf_varmeas.unique().tolist()
vlist

# for var in vlist:
#     print('\n\n')
#     s2 = s[s.puf_varmeas==var]
#     print(s2)

# pick one of the following 2 file names
# fname = r'C:\Users\donbo\Google Drive\NY PUF project\irs_puf_compare.txt'
date_id = date.today().strftime("%Y-%m-%d")
fbase = 'irs_puf_compare_' + date_id + '.txt'
# fr"{fbase}"
fname = r'C:\programs_python\puf_analysis\results\/' + fbase
#  + fr"{fbase}"

tfile = open(fname, 'a')
tfile.truncate(0)
# first write a summary with stub 0 for all variables
tfile.write('\n\n')
tfile.write('Summary report:\n')
tfile.write('  puf.csv advanced to 2017 with stage 1 and stage 2 only\n')
tfile.write('  filers only, using requirement rules and likely information\n\n')
s2 = s[s.common_stub==0]
tfile.write(s2.to_string())
# now write details for each variable
tfile.write('\n\nDetails by AGI range:')
for var in vlist:
    tfile.write('\n\n')
    s2 = s[s.puf_varmeas==var]
    tfile.write(s2.to_string())
tfile.close()


# %% OLD BELOW HERE develop usable targets

# drop targets for which I haven't yet set column descriptions as we won't
# use them
mask = irstot.variable.str.len() <= 2  # Excel column names will have length 2
irstot = irstot[~mask]
irstot = irstot.dropna(axis=0, subset=['column_description'])
irstot
irstot.columns

# check counts
irstot[['src', 'variable', 'table_description', 'value']].groupby(['src', 'table_description', 'variable']).agg(['count'])
vars = irstot[['variable', 'value']].groupby(['variable']).agg(['count'])  # unique list

# quick check to make sure duplicate variables have same values
# get unique combinations of src, variable
check = irstot[irstot.common_stub == 0][['src', 'variable']]
# indexes of duplicated combinations
idups = check.duplicated(subset='variable', keep=False)
check[idups].sort_values(['variable', 'src'])
dupvars = check[idups]['variable'].unique()
dupvars

# now check the stub 0 values of the variables that have duplicated values
qx = 'variable in @dupvars and common_stub==0'
vars = ['variable', 'column_description', 'src', 'value']
irstot.query(qx)[vars].sort_values(['variable', 'src'])
# looks ok except for very minor taxac differences
# any target version should be ok


# %% crosswalks on the fly
# dictionary xwalks between target name and puf name, AFTER constructing
# variables as needed in targets and in puf (as noted below)
TARGPUF_XWALK = dict(nret_all='nret_all',  # Table 1.1
                     # puf create nret_mars2, nret_mars1
                     nret_mfjss='nret_mars2',  # Table 1.2
                     nret_single='nret_mars1',  # Table 1.2
                     agi='c00100',  # Table 1.1
                     wages='e00200',  # Table 1.4
                     taxint='e00300',  # Table 1.4
                     orddiv='e00600',  # Table 1.4
                     # target cgnet = cggross - cgloss   # Table 1.4
                     cgnet='c01000',  # create cgnet in targets
                     # puf irapentot = e01400 + e01500 (taxable)
                     irapentot='irapentot',  # irapentot create in puf
                     socsectot='e02400',  # Table 1.4 NOTE THAT this is 'e'
                     ti='c04800'  # Table 1.1
                     )
TARGPUF_XWALK
# CAUTION: reverse xwalk relies on having only one keyword per value
PUFTARG_XWALK = {val: kw for kw, val in TARGPUF_XWALK.items()}


# %% prepare targets for comparison based on xwalks above
# define target variables
tlist = list(TARGPUF_XWALK.keys())  # values we want
# compute  cgnet = cggross - cgloss   # Table 1.4
tlist.remove('cgnet')
tlist.append('cggross')
tlist.append('cgloss')
tlist

# get the proper data
irstot
qx1 = 'variable in @tlist and '
qx2 = '((variable not in @dupvars) or '
qx3 = '(variable in @dupvars and src=="17in11si.xls"))'
qx = qx1 + qx2 + qx3
qx
vars = ['variable', 'common_stub', 'value']
target_base = irstot.query(qx)[vars]
target_base[['variable', 'value']].groupby(['variable']).agg(['count'])
# good, this is what we want

wide = target_base.pivot(index='common_stub', columns='variable', values='value')
# multiply the dollar-valued columns by 1000 (i.e., the non-num cols)
numcols = ['nret_all', 'nret_mfjss', 'nret_single']
dollcols = np.setdiff1d(wide.columns, numcols)
dollcols
wide[dollcols] = wide[dollcols] * 1000
wide['cgnet'] = wide['cggross'] - wide['cgloss']
wide = wide.drop(['cggross', 'cgloss'], axis=1)
wide['common_stub'] = wide.index
wide.columns
targets_long = pd.melt(wide, id_vars=['common_stub'])
targets_long['variable'].value_counts()


# alternative: here is the numpy equivalent to R ifelse
# targets_long['value'] = np.where(condition, targets_long['value'] * 1000, targets_long['value'])



