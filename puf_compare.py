# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 04:19:27 2020

@author: donbo
"""

# %% imports
import pandas as pd
import numpy as np
import puf_constants as pc
from datetime import date
import json

import puf_utilities as pu


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% constants
# pc.HT2_AGI_STUBS
# pc.ht2stubs
# pc.IRS_AGI_STUBS
# pc.irsstubs
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% functions

# prepare comp file and target_mappings
def prepall(puf, targets_possible):

    # setup
    target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
    target_vars = target_mappings.pufvar.to_list()

    # prepare puf
    # get names of puf variables for which we will need to create nnz indicator
    innz = target_mappings.pufvar.str.contains('_nnz')
    nnz_vars = target_mappings.pufvar[innz]
    pufvars_to_nnz = nnz_vars.str.rsplit(pat='_', n=1, expand=True)[0].to_list()
    puf = pu.prep_puf(puf, pufvars_to_nnz)

    # prepare puf subset
    idvars = ['pid', 'filer', 'common_stub', 's006']
    keep_vars = idvars + target_vars
    pufsub = puf.loc[puf['filer'], keep_vars]
    pufsub.columns.to_list()

    # prepare puf sums
    puflong = pufsub.melt(id_vars=idvars, var_name='pufvar')
    puflong['puf'] = puflong.s006 * puflong.value
    pufsums = puflong.groupby(['common_stub', 'pufvar'])[['puf']].sum().reset_index()

    grand_sums = pufsums.groupby(['pufvar']).sum().reset_index()
    grand_sums['common_stub'] = 0
    pufsums = pufsums.append(grand_sums)

    # prepare comp file
    comp = pd.merge(targets_possible.rename(columns={'value': 'irs'}),
                    pufsums, on=['common_stub', 'pufvar'])
    return comp, target_mappings


# comparison report
def comp_report(comp, outfile, title, target_mappings):
    target_vars = target_mappings.pufvar.to_list()
    comp['diff'] = comp['puf'] - comp['irs']
    comp['pdiff'] = comp['diff'] / comp['irs'] * 100
    # slim the file down
    mainvars = ['common_stub', 'incrange', 'pufvar', 'irsvar',
                'irs', 'puf', 'diff', 'pdiff', 'column_description']
    comp = comp[mainvars]
    # already should be sorted by pufvar dictionary order (pd.Categorical)
    # comp.sort_values(by=['irsvar', 'common_stub'], axis=0, inplace=True)

    s = comp.copy()
    # define custom sort order
    # s['pufvar'] = pd.Categorical(s['pufvar'], categories=pufirs_fullmap.keys(), ordered=True)
    # s = s.sort_values(by=['pufvar', 'common_stub'])

    s['pdiff'] = s['pdiff'] / 100.0
    format_mapping = {'irs': '{:,.0f}',
                      'puf': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}'}
    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    tfile.write('Data are for filers only, using IRS filing requirements plus estimates of likely filing.\n')
    tfile.write('\nThis report is in 3 sections:\n')
    tfile.write('  1. Summary report for all variables, summarized over all filers\n')
    tfile.write('  2. Detailed report by AGI range for each variable\n')
    tfile.write('  3. Table that provides details on puf variables and their mappings to irs data\n')
    tfile.write('\n1. Summary report for all variables, summarized over all filers:\n\n')
    s2 = s[s.common_stub==0]
    tfile.write(s2.to_string())
    # now write details for each variable
    tfile.write('\n\n2. Detail report by AGI range for each variable:')
    for var in target_vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(target_mappings.to_string())
    tfile.close()

    return


# %% get puf-variables and irs-variables linkages
pufirs_fullmap = json.load(open(DATADIR + 'pufirs_fullmap.json'))

# CAUTION: reverse xwalk relies on having only one keyword per value
irspuf_fullmap = {val: kw for kw, val in pufirs_fullmap.items()}


# %% get possible targets
targets_possible = pd.read_csv(DATADIR + 'targets_possible.csv')
# target_mappings = pd.read_csv(DATADIR + 'target_mappings.csv')
# target_vars = target_mappings.pufvar.to_list()


# %% get puf file(s)
puf_default = pd.read_parquet(PUF_DEFAULT, engine='pyarrow')
puf_regrown = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
# puf_default.columns.sort_values().to_list()


# %% compare and write results for default puf
comp, target_mappings = prepall(puf_default, targets_possible)

title_default = 'REPORT: puf.csv advanced to 2017, all 3 stages, tax-calculator growfactors and weights'
fname_default = RESULTDIR + 'irs_pufdefault_comparison.txt'

comp_report(comp,
            outfile=fname_default,
            title=title_default,
            target_mappings=target_mappings)


# %% compare and write results for regrown puf
comp, target_mappings = prepall(puf_regrown, targets_possible)

title_regrown = 'REPORT: puf.csv with custom stage1 growfactors, advanced to 2017 with tax-calculator stage 2 weights, no stage3'
fname_regrown = RESULTDIR + 'irs_pufregrown_comparison.txt'

comp_report(comp,
            outfile=fname_regrown,
            title=title_regrown,
            target_mappings=target_mappings)


