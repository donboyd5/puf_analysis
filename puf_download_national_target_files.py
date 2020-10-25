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


# %% ONETIME: collapse data to a common set of income ranges

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

# df['col1'].map(di)
# define a dictionary that maps the two income range definitions to a common definition
YEAR = '2017'
targets_all = pd.read_csv(DATADIR + 'targets' + YEAR + '.csv')

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

# df['col1'].map(di)
# define a dictionary that maps the two income range definitions to a common definition

# z['c'] = z.apply(lambda row: 0 if row['b'] in (0,1) else row['a'] / math.log(row['b']), axis=1)
# df['desired_output'] = df['data'].apply(lambda x: 'true' if x <= 2.5 else 'false')

# di = {}

# SLOW -- what is better
def f_nonitem(row):
    if row['irsstub'] == 0:
        return 0
    elif row['irsstub'] <= 2:
        return 1
    else:
        return row['irsstub'] - 1

def f_item(row):
    if row['irsstub'] <= 6:
        return row['irsstub']
    elif row['irsstub'] <= 8:
        return 7
    elif row['irsstub'] <= 10:
        return 8
    elif row['irsstub'] <= 13:
        return 9
    else:
        return row['irsstub'] - 4

# redefine the stubs
targets_collapsed = targets_all.copy()
targets_collapsed['common_stub'] = -99
item_mask = targets_collapsed.table_description.str.contains('Table 2.1')
targets_collapsed.loc[~item_mask, 'common_stub'] = targets_collapsed.apply(f_nonitem, axis=1)
targets_collapsed.loc[item_mask, 'common_stub'] = targets_collapsed.apply(f_item, axis=1)

# check
check = targets_collapsed.copy()
check['type'] = 'non_item'
check.loc[item_mask, 'type'] = 'item'
# check = check[['common_stub', 'type', 'irsstub', 'value']].groupby(['common_stub', 'type', 'irsstub']).agg(['count'])
check = check[['type', 'irsstub', 'common_stub', 'value']].groupby(['type', 'irsstub', 'common_stub']).agg(['count'])

# collapsed
targets_collapsed.columns
aggcols = ['src', 'common_stub', 'variable', 'table_description', 'column_description']
keepcols = aggcols + ['value']
targets_collapsed = targets_collapsed.groupby(aggcols)['value'].sum().reset_index()

# get the income range labels, reorder columns, and save
common_stub
targets_collapsed = pd.merge(targets_collapsed, common_stub, on=['common_stub'])
savecols = ['src', 'common_stub', 'incrange', 'variable', 'value', 'table_description', 'column_description']
targets_collapsed = targets_collapsed[savecols]
targets_collapsed.to_csv(DATADIR + 'targets' + YEAR + '_collapsed.csv', index=False)


# %% test parsing

# are reported totals close enough to sums of values that we can drop reported?
# df.iloc[1:, 1:].sum()
# df.iloc[0]
# df.iloc[1:, 1:].sum() - df.iloc[0]  # yes


