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

# temp = ht2[['ht2var', 'ht2description', 'pufvar']].drop_duplicates()

# %% imports
import requests
import pandas as pd
from io import StringIO

import json
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


# %% ONETIME: read Historical Table 2

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


# %% create new variables as needed
# iitax
#    A10300 total tax liability, minus
#      A85530 additional medicare tax
#      A09400 self-employment tax
#      refundable credits:
    #    A59720 excess EITC
    #    A11070 additional child credit
    #    A10960 refundable education credit
    #    A11560 net premium tax credit
# is a good approximation to the Tax-Calculator concept of iitax

# note, regarding business-like income and losses:
    # HT2 does have A00900 and N00900, but
    # does not have the counterpart to e02000 Rents/royalties, parternship/S, estate/trust
    # but it does have A26270 Partnership/S-corp net income (less loss) amount and N26270


ht2['Aiitax'] = (ht2.A10300 - ht2.A85530 - ht2.A09400
                 - (ht2.A59720 + ht2.A11070 + ht2.A10960 + ht2.A11560))

ht2['Niitax'] = ht2.N10300  # N must be at least this large -- we will ignore


# %% ONETIME: get variable labels
fn = 'ht2_variable_labels.xlsx'
vlabs = pd.read_excel(HT2DIR + fn, sheet_name = '2017')
vlabs['variable'] = vlabs.variable.str.lower()
vlabs
vlabs[vlabs.duplicated()]
vlabs.info()


# add vlabs for any variables just created using lowercase variable names
vlabs_new = {'variable': ['aiitax', 'niitax'],
             'description': ['Income taxes minus refundable credits, corresponds closely to Tax-Calculator iitax',
                             'IGNORE: Number of iitax returns, very rough estimate'],
             'type': ['Num', 'Num']}
vlabs_new_df = pd.DataFrame.from_dict(vlabs_new)

vlabs = pd.concat([vlabs, vlabs_new_df]).reset_index(drop=True)


# %% ONETIME: create long dataframe and save the file

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
dfl = dfl.rename(columns={'variable': 'ht2var', 'description': 'ht2description', 'value': 'ht2'})
dfl.to_csv(DATADIR + 'ht2_long_unmapped.csv', index=False)


# %% map ht2 variable names to puf names and save mappings

# create a dict that maps ht2 variable names (keys) to pufvars (values)
# we will build this from a set of dicts, as follows:
    # vars_dict has variables that must be mapped laboriously, ht2 variable to puf variable
    # vars_bar will have variables where the ht2 names and puf names are the same
    # vars ennz has variables where an ht2 variable corresponds to a puf "e" variable, and we also want the nnz variables
    # vars cnnz has variables where an ht2 variable corresponds to a puf "c" variable, and we also want the nnz variables

vars_dict = {'n1': 'nret_all', 'aiitax': 'iitax', 'niitax': 'iitax_nnz'}
vars_bare = ('mars1', 'mars2', 'mars4')

vars_ennz = ('00200', '00300', '00600', '01500', '02400')
vars_cnnz = ('00100', '01000', '02500', '04800', '05800', '17000', '18300', '19200', '19700')

# posneg vars
# c01000pos
# e26270pos

dict_bare = {}
for keyval in vars_bare:
    dict_bare[keyval] = keyval

dict_ennz = {}
for ennz in vars_ennz:
    key = 'a' + ennz
    value = 'e' + ennz
    dict_ennz[key] = value
    key = 'n' + ennz
    value = 'e' + ennz + '_nnz'
    dict_ennz[key] = value

dict_cnnz = {}
for cnnz in vars_cnnz:
    key = 'a' + cnnz
    value = 'c' + cnnz
    dict_cnnz[key] = value
    key = 'n' + cnnz
    value = 'c' + cnnz + '_nnz'
    dict_cnnz[key] = value

ht2puf_map = {}
ht2puf_map.update(vars_dict)
ht2puf_map.update(dict_bare)
ht2puf_map.update(dict_ennz)
ht2puf_map.update(dict_cnnz)

ht2puf_map

json.dump(ht2puf_map, open(DATADIR + 'ht2puf_fullmap.json', 'w'))


# %% map and save

dfl_mapped = pd.read_csv(DATADIR + 'ht2_long_unmapped.csv')
dfl_mapped
sorted(dfl_mapped.ht2var.unique())

dfl_mapped['pufvar'] = dfl_mapped.ht2var.map(ht2puf_map)
dfl_mapped.to_csv(DATADIR + 'ht2_long.csv', index=False)

