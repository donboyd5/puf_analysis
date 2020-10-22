# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:51:44 2020

@author: donbo
"""
# %% imports
import requests
import pandas as pd


# %% constants
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
PUFDIR = 'C:/programs_python/weighting/puf/'


# %% ONETIME:  IRS Table urls
# Main url: https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income

# Tables from SOI Individual Complete Report (Publication 1304)

# Category 1: Individual Income Tax Returns Filed and Sources of Income

#  Table 1.1 Selected Income and Tax Items
#  By Size and Accumulated Size of Adjusted Gross Income
TAB11 = '18in11si.xls'
TAB11d = {'src': '18in11si.xls',
          'firstrow': 10,
          'lastrow': 29,
          'cols': 'A, B, D, K, L',
          'colnames': ['incrange', 'nret_all', 'agi', 'nret_ti', 'ti']}

#  Table 1.2 Adjusted Gross Income, Exemptions, Deductions, and Tax Items
#  By Size of Adjusted Gross Income and Marital Status
TAB12 = '18in12ms.xls'

#  Table 1.4 Sources of Income, Adjustments Deductions and Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB14 = '18in14ar.xls'

#  Table 1.4A Returns with Income or Loss from Sales of Capital Assets Reported on Form1040, Schedule D
#  By Size of Adjusted Gross Income
TAB14A = '18in14acg.xls'

#  Table 1.6 Number of Returns
#  By Size of Adjusted Gross Income, Marital Status, and Age of Taxpayer
TAB16 = '18in16ag.xls'

# Category 2: Individual Income Tax Returns with Exemptions and Itemized Deductions

#  Table 2.1 Individual Income Tax Returns with Itemized Deductions:
#  Sources of Income, Adjustments, Itemized Deductions by Type, Exemptions, and Tax Items
#  By Size of Adjusted Gross Income
TAB21 = '18in21id.xls'

#  Table 2.5 Individual Income Tax Returns with Earned Income Credit
#  By Size of Adjusted Gross Income
#  https://www.irs.gov/pub/irs-soi/18in25ic.xls
TAB25 = '18in25ic.xls'

# Category 3: Individual Income Tax Returns with Tax Computation
#  Table 3.2 Individual Income Tax Returns with Total Income Tax:
#  Total Income Tax as a Percentage of Adjusted Gross Income
TAB32 = '18in32tt.xls'

files = [TAB11, TAB12, TAB14, TAB14A, TAB16, TAB21, TAB25, TAB32]


# %% ONETIME:  download and save files

for f in files:
    print(f)
    url = WEBDIR + f
    path = DOWNDIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


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


# %% parse and save important file contents
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html

# get the names and info for tables we want data from
fn = r'C:\programs_python\weighting\puf\data\soitables.xlsx'
tabs = pd.read_excel(io=fn, sheet_name='national')
tabmaps = pd.read_excel(io=fn, sheet_name='tablemaps')

# loop through the tables listed in tabs
# tab = 'tab14'
tabs.table
tabsuse = tabs.table.drop([3])  # not ready to use tab21

tablist = []
for tab in tabsuse:
    # get info describing a specific table
    tabd = tabs[tabs['table'] == tab]
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
targets_all.to_csv(DATADIR + 'targets2018.csv', index=False)

# note that we can drop targets not yet identified by dropping those where
# column_description is NaN (or where len(variable) <= 2)


# %% test parsing

# are reported totals close enough to sums of values that we can drop reported?
# df.iloc[1:, 1:].sum()
# df.iloc[0]
# df.iloc[1:, 1:].sum() - df.iloc[0]  # yes


