# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:51:44 2020

@author: donbo
"""
# %% imports
import requests
import pandas as pd
from io import StringIO


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'  # static files
DOWNDIR = 'C:/programs_python/puf_analysis/downloads/'
DATADIR = 'C:/programs_python/puf_analysis/data/'
# PUFDIR = 'C:/programs_python/weighting/puf/'


# %% functions


# %% xlrange
def xlrange(io, sheet_name=0,
            firstrow=1, lastrow=None,
            usecols=None, colnames=None):
    # firstrow and lastrow are 1-based
    if colnames is None:
        if usecols is None:
            colnames = None
        elif isinstance(usecols, list):
            colnames = usecols
        else:
            colnames = usecols.split(',')
    nrows = None
    if lastrow is not None:
        nrows = lastrow - firstrow + 1
    df = pd.read_excel(io,
                       header=None,
                       names=colnames,
                       usecols=usecols,
                       skiprows=firstrow - 1,
                       nrows=nrows)
    return df


# %% ONETIME:  2017 IRS Table urls
# Main url: https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income

# Tables from SOI Individual Complete Report (Publication 1304)

# Category 1: Individual Income Tax Returns Filed and Sources of Income

#  Table 1.1 Selected Income and Tax Items
#  By Size and Accumulated Size of Adjusted Gross Income
TAB11_2017 = '17in11si.xls'

#  Table 1.2 Adjusted Gross Income, Exemptions, Deductions, and Tax Items
#  By Size of Adjusted Gross Income and Marital Status
TAB12_2017 = '17in12ms.xls'

#  Table 1.4 Sources of Income, Adjustments Deductions and Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB14_2017 = '17in14ar.xls'

#  Table 1.4A Returns with Income or Loss from Sales of Capital Assets Reported on Form1040, Schedule D
#  By Size of Adjusted Gross Income
TAB14A_2017 = '17in14acg.xls'

#  Table 1.6 Number of Returns
#  By Size of Adjusted Gross Income, Marital Status, and Age of Taxpayer
TAB16_2017 = '17in16ag.xls'

# Category 2: Individual Income Tax Returns with Exemptions and Itemized Deductions

#  Table 2.1 Individual Income Tax Returns with Itemized Deductions:
#  Sources of Income, Adjustments, Itemized Deductions by Type, Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB21_2017 = '17in21id.xls'

#  Table 2.5 Individual Income Tax Returns with Earned Income Credit
#  By Size of Adjusted Gross Income
#  https://www.irs.gov/pub/irs-soi/18in25ic.xls
TAB25_2017 = '17in25ic.xls'

# Category 3: Individual Income Tax Returns with Tax Computation
#  Table 3.2 Individual Income Tax Returns with Total Income Tax:
#  Total Income Tax as a Percentage of Adjusted Gross Income
TAB32_2017 = '17in32tt.xls'

files_2017 = [TAB11_2017, TAB12_2017, TAB14_2017, TAB14A_2017,
              TAB16_2017, TAB21_2017, TAB25_2017, TAB32_2017]


# %% ONETIME:  2018 IRS Table urls
# Main url: https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income

# Tables from SOI Individual Complete Report (Publication 1304)

# Category 1: Individual Income Tax Returns Filed and Sources of Income

#  Table 1.1 Selected Income and Tax Items
#  By Size and Accumulated Size of Adjusted Gross Income
TAB11_2018 = '18in11si.xls'
TAB11d = {'src': '18in11si.xls',
          'firstrow': 10,
          'lastrow': 29,
          'cols': 'A, B, D, K, L',
          'colnames': ['incrange', 'nret_all', 'agi', 'nret_ti', 'ti']}

#  Table 1.2 Adjusted Gross Income, Exemptions, Deductions, and Tax Items
#  By Size of Adjusted Gross Income and Marital Status
TAB12_2018 = '18in12ms.xls'

#  Table 1.4 Sources of Income, Adjustments Deductions and Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB14_2018 = '18in14ar.xls'

#  Table 1.4A Returns with Income or Loss from Sales of Capital Assets Reported on Form1040, Schedule D
#  By Size of Adjusted Gross Income
TAB14A_2018 = '18in14acg.xls'

#  Table 1.6 Number of Returns
#  By Size of Adjusted Gross Income, Marital Status, and Age of Taxpayer
TAB16_2018 = '18in16ag.xls'

# Category 2: Individual Income Tax Returns with Exemptions and Itemized Deductions

#  Table 2.1 Individual Income Tax Returns with Itemized Deductions:
#  Sources of Income, Adjustments, Itemized Deductions by Type, Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB21_2018 = '18in21id.xls'

#  Table 2.5 Individual Income Tax Returns with Earned Income Credit
#  By Size of Adjusted Gross Income
#  https://www.irs.gov/pub/irs-soi/18in25ic.xls
TAB25_2018 = '18in25ic.xls'

# Category 3: Individual Income Tax Returns with Tax Computation
#  Table 3.2 Individual Income Tax Returns with Total Income Tax:
#  Total Income Tax as a Percentage of Adjusted Gross Income
TAB32_2018 = '18in32tt.xls'

files_2018 = [TAB11_2018, TAB12_2018, TAB14_2018, TAB14A_2018, TAB16_2018,
              TAB21_2018, TAB25_2018, TAB32_2018]


# %% ONETIME:  download and save files

files = files_2017

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% parse and save important file contents
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html

# get the names and info for tables we want data from
YEAR = '2017'

fn = r'C:\programs_python\puf_analysis\data\soitables.xlsx'
tabs = pd.read_excel(io=fn, sheet_name='national_' + YEAR)
tabmaps = pd.read_excel(io=fn, sheet_name='tablemaps_' + YEAR)

# loop through the tables listed in tabs
# tab = 'tab14'
tabs.table
# tabsuse = tabs.table.drop([3])  # not ready to use tab21
tabsuse = tabs.table

tablist = []
tab = tabsuse[3]
for tab in tabsuse:
    # get info describing a specific table
    tabd = tabs[tabs['table'] == tab]  # df row with table description
    tabinfo = pd.merge(tabd, tabmaps, on='table')

    # get the table data using this info
    df = xlrange(io=DOWNDIR + tabd.src.values[0],
                 firstrow=tabd.firstrow.values[0],
                 lastrow=tabd.lastrow.values[0],
                 usecols=tabinfo.col.str.cat(sep=", "),
                 colnames=tabinfo.colname.tolist())

    # add identifiers
    df['src'] = tabd.src.values[0]
    df['irsstub'] = df.index

    # melt to long format so that all data frames are in same format
    dfl = pd.melt(df, id_vars=['src', 'irsstub', 'incrange'])

    # bring table description and column description into the table
    dfl = pd.merge(dfl,
                   tabinfo[['colname', 'table_description', 'column_description']],
                   left_on=['variable'],
                   right_on=['colname'])
    dfl = dfl.drop('colname', axis=1)  # colname duplicates variable so drop it

    # add to the list
    tablist.append(dfl)

targets_all = pd.concat(tablist)
targets_all.info()
targets_all.to_csv(DATADIR + 'targets' + YEAR + '.csv', index=False)

# note that we can drop targets not yet identified by dropping those where
# column_description is NaN (or where len(variable) <= 2)


# %% ONETIME save irs income range mappings based on data
# create irsstub and incrange mapping
# incrange for irsstub 0 and 1 doesn't have consistent text values so set them

# ranges for 2017 and 2018 are the same, but if additional years are added
# always verify that ranges are the same

# irs uses two groupings: 
#   19 ranges plus a total, for most files
#   22 ranges plus a total, for itemized deductions; we will create a crosswalk

# targets_all = pd.read_csv(DATADIR + 'targets' + YEAR + '.csv')
# targets_all.table_description.values

# define mask to identify rows from the itemized deduction table
item_mask = targets_all.table_description.str.contains('Table 2.1')

# first, get our main set of ranges (19 + )
non_item = targets_all[~item_mask].copy()
# verify
non_item.groupby(['table_description'])['table_description'].count()

# check for range name conformity and create a single set of names
non_item.groupby(['incrange'])['incrange'].count()
non_item.loc[non_item['irsstub'] == 0, 'incrange'] = 'All returns'
non_item.loc[non_item['irsstub'] == 1, 'incrange'] = 'No adjusted gross income'
non_item['incrange'] = non_item['incrange'].str.strip()
incmap = non_item[['irsstub', 'incrange']].drop_duplicates()
incmap
incmap.to_csv(DATADIR + 'irsstub_labels.csv', index=False)

# second, get the itemized deduction ranges
item = targets_all[item_mask].copy()
# verify
item.groupby(['table_description'])['table_description'].count()

# check for range name conformity and create a single set of names
item.groupby(['incrange'])['incrange'].count()
# the names conform so we are all set
item.loc[item['irsstub'] == 0, 'incrange'] = 'All returns'
item['incrange'] = item['incrange'].str.strip()
idincmap = item[['irsstub', 'incrange']].drop_duplicates()
idincmap
idincmap.to_csv(DATADIR + 'irsstub_itemded_labels.csv', index=False)


# %% ONETIME: save common_stub labels

# common_stub	irs stubs	item stubs
# 0	0	0
# 1	1, 2	1
# 2	3	2
# 3	4	3
# 4	5	4
# 5	6	5
# 6	7	6
# 7	8	7 ,8
# 8	9	9, 10
# 9	10	11, 12, 13
# 10	11	14
# 11	12	15
# 12	13	16
# 13	14	17
# 14	15	18
# 15	16	19
# 16	17	20
# 17	18	21
# 18	19	22

STUB_DATA = '''common_stub;incrange
0; All returns
1; Under $5,000
2; $5,000 under $10,000
3; $10,000 under $15,000
4; $15,000 under $20,000
5; $20,000 under $25,000
6; $25,000 under $30,000
7; $30,000 under $40,000
8; $40,000 under $50,000
9; $50,000 under $75,000
10; $75,000 under $100,000
11; $100,000 under $200,000
12; $200,000 under $500,000
13; $500,000 under $1,000,000
14; $1,000,000 under $1,500,000
15; $1,500,000 under $2,000,000
16; $2,000,000 under $5,000,000
17; $5,000,000 under $10,000,000
18; $10,000,000 or more
'''

common_stub = pd.read_table(StringIO(STUB_DATA), sep=';')
common_stub['incrange'] = common_stub['incrange'].str.strip()

common_stub.to_csv(DATADIR + 'irsstub_common_labels.csv', index=False)


# %% ONETIME: clean data and collapse to the common set of income ranges
# see https://kanoki.org/2019/04/06/pandas-map-dictionary-values-with-dataframe-columns/

# get targets, delete duplicates, remap stubs, drop unneeded columns
# remap the stubs so that we can collapse
YEAR = '2017'
targets_remap = pd.read_csv(DATADIR + 'targets' + YEAR + '.csv').drop(['incrange'], axis=1)
targets_remap.columns
targets_remap.info()
targets_remap['value'] = pd.to_numeric(targets_remap['value'], errors='coerce')
targets_remap.value.max()

# drop targets for which I haven't yet set column descriptions as we won't
# use them
mask = targets_remap.variable.str.len() <= 2  # Excel column names will have length 2
targets_remap = targets_remap[~mask]
targets_remap = targets_remap.dropna(axis=0, subset=['column_description'])
targets_remap
targets_remap.columns

# create dicts to redefine stubs, keys are non-id or id stubs, values are common stubs
# non-itemized mapping
keys = range(0, 20)
values = range(-1, 19)  # for income stubs, default common is 1 less than inc stub
nonid_map = dict(zip(keys, values))
nonid_map[0] = 0
nonid_map[1] = 1
nonid_map[2] = 1
nonid_map

# itemized mapping
keys = range(0, 23)
values = tuple(range(0, 7)) + (7, 7, 8, 8, 9, 9, 9) + tuple(range(10, 19))
values
id_map = dict(zip(keys, values))
id_map

# identifier for itemized-deduction table, which has different stubs
item_mask = targets_remap.table_description.str.contains('Table 2.1')
targets_remap.loc[~item_mask, 'common_stub'] = targets_remap.irsstub.map(nonid_map)
targets_remap.loc[item_mask, 'common_stub'] = targets_remap.irsstub.map(id_map)
targets_remap = targets_remap.astype({'common_stub': int})

# check
check = targets_remap.copy()
check['type'] = 'non_item'
check.loc[item_mask, 'type'] = 'item'
# check = check[['common_stub', 'type', 'irsstub', 'value']].groupby(['common_stub', 'type', 'irsstub']).agg(['count'])
check = check[['type', 'irsstub', 'common_stub', 'value']].groupby(['type', 'irsstub', 'common_stub']).agg(['count'])
check

# drop duplicate records, unnecessary columns, and collapse
targets_remap.columns

# quick check to make sure duplicate variables have same values
# get unique combinations of src, variable
check = targets_remap[targets_remap.common_stub == 0][['src', 'variable']]
# indexes of duplicated combinations
idups = check.duplicated(subset='variable', keep=False)
check[idups].sort_values(['variable', 'src'])
dupvars = check[idups]['variable'].unique()
dupvars

# now check the stub 0 values of the variables that have duplicated values
qx = 'variable in @dupvars and common_stub==0'
vars = ['variable', 'column_description', 'src', 'value']
targets_remap.query(qx)[vars].sort_values(['variable', 'src'])
# looks ok except for very minor taxac differences
# any target version should be ok

# get unduplicated data
qx1 = '((variable not in @dupvars) or '
qx2 = '(variable in @dupvars and src=="17in11si.xls"))'
qx = qx1 + qx2
qx
vars = ['variable', 'common_stub', 'value']
targets_undup = targets_remap.query(qx)
targets_undup[['variable', 'value']].groupby(['variable']).agg(['count'])

# collapse without irsstub
# data.groupby('month')[['duration']].sum()
aggcols = ['src', 'common_stub', 'variable', 'table_description', 'column_description']
targets_collapsed = targets_undup.groupby(aggcols)[['value']].sum().reset_index()  # [[]] keeps data frame
targets_collapsed.columns

# get the income range labels, reorder columns, put dollars in thousands, and save
common_stub_labels = pd.read_csv(DATADIR + 'irsstub_common_labels.csv')
targets_collapsed = pd.merge(targets_collapsed, common_stub_labels, on=['common_stub'])
savecols = ['src', 'common_stub', 'incrange', 'variable', 'value', 'table_description', 'column_description']
targets_collapsed = targets_collapsed[savecols]

# multiply dollar amounts by 1000
dollar_xmask = targets_collapsed.variable.str.contains('nret_|n_')
dollar_xmask.sum()
dollar_xmask[range(26, 30)]
targets_collapsed.loc[~dollar_xmask, 'value'] = targets_collapsed.loc[~dollar_xmask, 'value'] * 1000
targets_collapsed.to_csv(DATADIR + 'targets' + YEAR + '_collapsed.csv', index=False)


