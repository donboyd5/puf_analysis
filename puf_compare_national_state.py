# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 05:56:47 2020

@author: donbo
"""


# %% imports
import requests
import pandas as pd
import numpy as np
from io import StringIO
import json

import puf_constants as pc
import puf_utilities as pu


# %% locations and file names
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
DOWNDIR = IGNOREDIR + 'downloads/'
HT2DIR = IGNOREDIR + 'Historical Table 2/'
DATADIR = 'C:/programs_python/puf_analysis/data/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'


# %% get national and state files
# get previously determined national-puf mapping
# json.dump(pufirs_fullmap, open(DATADIR + 'pufirs_fullmap.json', 'w'))
# this dict defines the order in which we want tables sorted, so get it
pufirs_fullmap = json.load(open(DATADIR + 'pufirs_fullmap.json'))
# or just use pc.pufirs_fullmap

targets_national = pd.read_csv(DATADIR + 'targets2017_possible.csv')
targets_ht2 = pd.read_csv(DATADIR + 'ht2_long.csv')

targets_national.columns
# targets_national.common_stub.value_counts().sort_values()
# targets_national.dtypes

pu.uvals(targets_national.irsvar)
pu.uvals(targets_national.pufvar)


# %% create mergeable files

# create common income range mapping between the national and state summary files
pc.ht2common_stubs  # map ht2stubs and common stubs to these stubs

# keys are the original values, values are the new values
# ht2 stubs are an easy mapping
keys = range(0, 11)
values = (0, 1, 1) + tuple(range(2, 10))
ht2_map = dict(zip(keys, values))

keys = range(0, 19)
values = (0, ) + (1,) * 2 + (2,) * 3 + (3,) * 3 + tuple(range(4, 9)) + (9,) * 5
irs_map = dict(zip(keys, values))
irs_map

# national drop vars, add ht2common_stub
targets_national.loc[:, ['ht2common_stub']] = targets_national.common_stub.map(irs_map)
aggcols = ['ht2common_stub', 'pufvar', 'irsvar', 'column_description', 'src', 'table_description', 'excel_column']
targets_national_mrg = targets_national.groupby(aggcols)[['irs']].sum().reset_index()
targets_national_mrg = targets_national_mrg.rename(columns={'column_description': 'irs_column',
                                                    'table_description': 'irs_table',
                                                    'src': 'irs_src',
                                                    'excel_column': 'irs_xlcolumn'})
targets_national_mrg.columns


# ht2 keep US, add ht2common_stub
targets_ht2.loc[:, ['ht2common_stub']] = targets_ht2.ht2_stub.map(ht2_map)
targets_ht2 = targets_ht2[targets_ht2.state=='US'].drop(columns=['state'])
targets_ht2
aggcols = ['ht2common_stub', 'pufvar', 'ht2var', 'ht2description']
targets_ht2_mrg = targets_ht2.groupby(aggcols)[['ht2']].sum().reset_index()
targets_ht2_mrg.columns


# %% merge
natstate = pd.merge(targets_national_mrg, targets_ht2_mrg,
                    how='inner', on=['ht2common_stub', 'pufvar'])
natstate = pd.merge(natstate, pc.ht2common_stubs, on=['ht2common_stub'])

natstate['diff'] = natstate.ht2 - natstate.irs
natstate['pdiff'] = natstate['diff'] / natstate.irs * 100
natstate['abspdiff'] = np.abs(natstate.pdiff)

# reorder variables
mainvars = ['ht2common_stub', 'incrange', 'irsvar', 'ht2var', 'pufvar',
            'irs', 'ht2', 'diff', 'pdiff', 'abspdiff',
            'irs_column', 'ht2description', 'irs_table', 'irs_xlcolumn']
natstate = natstate[mainvars]
natstate.columns
natstate.pufvar.unique()
natstate.to_csv(DATADIR + 'irs_ht2_national_values.csv')


# %% functions
# comparison report
def natstate_report(natstate, outfile, title, pufirs_fullmap):
    comp = natstate.copy()
    # modify pdiff variables for printing purposes
    comp['pdiff'] = comp.pdiff / 100
    comp['abspdiff'] = comp.abspdiff / 100

    # sorted by pufvar dictionary order (pd.Categorical)
    comp['pufvar'] = pd.Categorical(comp['pufvar'], categories=pufirs_fullmap.keys(), ordered=True)
    comp = comp.sort_values(by=['pufvar', 'ht2common_stub'])

    reportvars = ['ht2common_stub', 'incrange', 'irsvar', 'ht2var', 'pufvar',
                'irs', 'ht2', 'diff', 'pdiff', 'abspdiff']

    vars = comp.pufvar.unique()

    idvars = ['irsvar', 'ht2var', 'pufvar',
                'irs_column', 'ht2description', 'irs_table', 'irs_xlcolumn']
    mappings = comp[idvars].drop_duplicates()

    s = comp.copy()
    s=s[reportvars]

    format_mapping = {'irs': '{:,.0f}',
                      'ht2': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}',
                      'abspdiff': '{:.1%}'}
    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n')
    tfile.write('\nThis report is in 3 sections:\n')
    tfile.write('  1. Summary report for all variabless\n')
    tfile.write('  2. Detailed report by AGI range for each variable\n')
    tfile.write('  3. Table that provides details on ht2 variables and their mappings to irs data\n')
    tfile.write('\n1. Summary report for all variables, summarized over all filers:\n\n')
    s2 = s[s.ht2common_stub==0]
    tfile.write(s2.to_string())
    # now write details for each variable
    tfile.write('\n\n2. Detailed report by AGI range for each variable:')
    for var in vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(mappings.to_string())
    tfile.close()

    return


# %% create report
rtitle = 'REPORT: Historical Table 2 data for nation compared to IRS national statistics, 2017'
fname = RESULTDIR + 'irs_ht2_comparison.txt'
natstate_report(natstate, outfile=fname, title=rtitle, pufirs_fullmap=pufirs_fullmap)

