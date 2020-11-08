# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 09:51:44 2020

@author: donbo
"""

# %% tasks
#  download historical table 2
#  read file, make long
#  convert variable names to lower case
#  convert string to numeric
#  convert from $k to dollars
#  add variable labels
#  save as csv

# %% notes and urls
# https://github.com/PSLmodels/Tax-Calculator


# %% imports
import requests
import pandas as pd
from io import StringIO

import puf_constants as pc


# %% locations and file names
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'  # static files
WEBHT2 = 'https://www.irs.gov/statistics/soi-tax-stats-historic-table-2'

IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
DOWNDIR = IGNOREDIR + 'downloads/'
HT2DIR = IGNOREDIR + 'Historical Table 2/'
DATADIR = 'C:/programs_python/puf_analysis/data/'
# PUFDIR = 'C:/programs_python/weighting/puf/'

HT2_2017 = '17in54cmcsv.csv'  # WEBDIR + HT2_2017
HT2_2018 = '18in55cmagi.csv'

# excel
# https://www.irs.gov/pub/irs-soi/17in54cm.xlsx

# documentation
# https://www.irs.gov/pub/irs-soi/17incmdocguide.doc
# https://www.irs.gov/pub/irs-soi/17in33ny.xlsx


# %% constants
pc.ht2stubs
pc.STATES


# %% functions


# %% ONETIME:  download Historical Table 2 files

# files = files_2017
files = [HT2_2017, '17incmdocguide.doc', '17in54cm.xlsx', '17in33ny.xlsx']

for f in files:
    print(f)
    url = WEBDIR + f
    path = HT2DIR + f
    r = requests.get(url)
    print(r.status_code)
    with open(path, "wb") as file:
        file.write(r.content)


# %% get variable labels
fn = 'ht2_variable_labels.xlsx'
vlabs = pd.read_excel(HT2DIR + fn, sheet_name = '2017')
vlabs['variable'] = vlabs.variable.str.lower()
vlabs


# %% read and adjust Historical Table 2

ht2 = pd.read_csv(HT2DIR + HT2_2017, thousands=',')
ht2
ht2.dtypes  # they are integer, we want floats
ht2.info()
ht2.STATE.describe()  # 54 -- states, DC, US, PR, OA
ht2.STATE.value_counts().sort_values()
ht2.groupby('STATE').STATE.count()  # alpha order
ht2.head()
ht2.columns.to_list()
# # convert all strings to numeric
# stn = ht2raw.columns.to_list()
# stn.remove('STATE')
# ht2[stn] = ht2raw[stn].apply(pd.to_numeric, errors='coerce', axis=1)
ht2

dfl = pd.melt(ht2, id_vars=['STATE', 'AGI_STUB'])
dfl.info
dfl.dtypes
dfl = dfl.rename(columns={'STATE': 'state', 'AGI_STUB': 'ht2_stub'})
dfl['value'] = dfl['value'].astype(float)
dfl.describe()
vars = dfl.variable.value_counts().reset_index()

# multiply dollar amounts by 1000
dollar_mask = dfl.variable.str.startswith('A')
dollar_mask.sum()
dfl.loc[dollar_mask, 'value'] *= 1000

# convert variable names to lower case
dfl['variable'] = dfl.variable.str.lower()
dfl

dfl = pd.merge(dfl, vlabs[['variable', 'description', 'reference']],
               how='left', on=['variable'])

check = dfl[['variable', 'description']].drop_duplicates()

dfl = dfl[['state', 'variable', 'description', 'ht2_stub', 'value']].sort_values(by=['state', 'variable', 'ht2_stub'])
dfl = dfl.rename(columns={'variable': 'ht2variable', 'description': 'ht2description', 'value': 'ht2'})

dfl.to_csv(DATADIR + 'ht2_long.csv', index=False)

