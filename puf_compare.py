# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 04:19:27 2020

@author: donbo
"""

# %% imports
import pandas as pd
import numpy as np
import puf_constants as pc


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
HDFDIR = r'C:\programs_python\puf_analysis\ignore/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'

BASE_NAME = 'puf_adjusted'
PUF_HDF = HDFDIR + BASE_NAME + '.h5'  # hdf5 is lightning fast


# %% constants

# pc.HT2_AGI_STUBS
# pc.ht2stubs
# pc.IRS_AGI_STUBS
# pc.irsstubs


# %% get target data
IRSDAT = DATADIR + 'targets2017_collapsed.csv'
irstot = pd.read_csv(IRSDAT)
irstot
irstot.info()
irstot.count()
irsvars = irstot.variable.value_counts().sort_index()


# %% get the puf
puf = pd.read_hdf(IGNOREDIR + 'puf2017_2020-10-26.h5')  # 1 sec
puf['common_stub'] = pd.cut(
    puf['c00100'],
    pc.COMMON_STUBS,
    labels=range(1, 19),
    right=False)
puf.info


# %% get nz counts and weighted sums of most puf variables
# get the subset of variables we want
pufvars = puf.columns
keepcols = ('pid', 'common_stub', 's006', 'c00100', 'e00200', 'e00300',
            'e00600', 'e01500')
pufsub = puf.loc[:, keepcols]

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

vmap = {'c00100_nnz': 'nret_all',
        'c00100_wsum': 'agi',
        'e00200_nnz': 'nret_wages',
        'e00200_wsum': 'wages',
        'e00300_nnz': 'nret_taxint',
        'e00300_wsum': 'taxint',
        'e00600_nnz': 'nret_orddiv',
        'e00600_wsum': 'orddiv',
        'e01500_nnz': 'nret_pensions',
        'e01500_wsum': 'pensions'}

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


# %% print or write results 
# ['src']
s = comp.copy()[mainvars + infovars]
s['pdiff'] = s['pdiff'] / 100.0
format_mapping = {'irs': '{:,.0f}',
                  'puf': '{:,.0f}',
                  'diff': '{:,.0f}',
                  'pdiff': '{:.1%}'}
for key, value in format_mapping.items():
    s[key] = s[key].apply(value.format)

vlist = comp.irsvar.unique().sort()
vlist = comp.puf_varmeas.unique().tolist()
vlist.sort()
vlist

for var in vlist:
    print('\n\n')
    s2 = s[s.puf_varmeas==var]
    print(s2)

tfile = open(r'c:\temp\irs_puf_compare.txt', 'a')
tfile.truncate(0)
for var in vlist:
    tfile.write('\n\n\n')
    s2 = s[s.puf_varmeas==var]
    tfile.write(s2.to_string())
tfile.close()


# %% develop usable targets

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



