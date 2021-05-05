# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 05:46:17 2020

@author: donbo
"""

# %% imports
import numpy as np
import pandas as pd

import puf_constants as pc


# %% functions


def comp_report(comp, outfile, title):

    # comparison report
    print(f'Preparing report...')

    # modify pufvar to allow sort by pufvar dictionary order (pd.Categorical)
    comp['pufvar'] = pd.Categorical(comp.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)

    state_groups = comp.stgroup.unique()

    comp.sort_values(by=['pufvar', 'stgroup', 'ht2_stub'], axis=0, inplace=True)
    target_vars = comp.pufvar.unique()

    # target_vars = comp.pufvar.unique()

    print(f'Writing report...')
    s = comp.copy()
    s['pdiff'] = s['pdiff'] / 100.0
    s['abspdiff'] = s['abspdiff'] / 100.0
    format_mapping = {'target': '{:,.0f}',
                      'puf': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}',
                      'abspdiff': '{:.1%}'}
    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    def qwrite(qx, qtitle, sort=['pufvar', 'ht2_stub', 'stgroup']):
        qtext = '\n\n    ' + qtitle + ':\n\n'
        tfile.write(qtext)
        s2 = s.query(qx).copy()
        s2.sort_values(by=sort, axis=0, inplace=True)
        tfile.write(s2.to_string())
        return

    def var_states(var):
        qtitle = '\nSummary for ' + var + ' for each state.\n'
        tfile.write(qtitle)
        for st in state_groups:
            qx = 'stgroup=="' + st + '" and pufvar=="' + var + '"'
            qtitle = 'Summary for ' + var + ' in ' + st
            qwrite(qx, qtitle, sort=['ht2_stub'])

    def state_vars(st):
        qtitle = '\nSummary for ' + st + ' of each variable.\n'
        tfile.write(qtitle)
        for var in target_vars:
            qx = 'stgroup=="' + st + '" and pufvar=="' + var + '"'
            qtitle = 'Summary for ' + var + ' in ' + st
            qwrite(qx, qtitle, sort=['ht2_stub'])

    tfile = open(outfile, 'a')
    tfile.truncate(0)

    tfile.write('\n' + title + '\n\n')
    tfile.write('Data are for filers only, using IRS filing requirements plus estimates of likely filing.\n\n')
    tfile.write('Note that comparisons are to puf-based state targets, not to actual Historical Table 2 targets:\n\n')
    tfile.write('  For example, if the HT2 U.S. total for a variable is $100 and the IRS national total for\n')
    tfile.write('  the variable is $95, then each state target will be a pro-rata share of the $95, based\n')
    tfile.write('  upon the state share of $100, and the comparison in the tables below is to the pro-rata\n')
    tfile.write('  share of $95.\n')

    tfile.write('\nThis report has the following sections:\n')
    tfile.write('  * Summary report for # of returns in each state\n')
    tfile.write('  * Summary report for all variables in New York\n')
    tfile.write('  * Details by income range for each variable in New York\n')
    tfile.write('  * Details by income range for selected variables in each state\n')
    tfile.write('    - nret_all\n')
    tfile.write('    - c00100\n')
    tfile.write('    - c04800\n')
    tfile.write('    - c05800\n')

    # tfile.write('\n')

    # s.sort_values(by=['pufvar', 'ht2_stub'], axis=0, inplace=True)
    qx = "pufvar=='nret_all' and ht2_stub==0"
    qtitle = 'Number of returns by state'
    qwrite(qx, qtitle, sort=['stgroup'])

    qx = "stgroup=='NY' and ht2_stub==0"
    qtitle = 'Summary for New York'
    qwrite(qx, qtitle, sort=['pufvar'])

    state_vars('NY')

    var_states('nret_all')
    var_states('c00100')
    var_states('c04800')
    var_states('c05800')

    # # finally, write the mapping
    # tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    # tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()

    return #  comp return nothing or return comp?


def get_onestate_wsums(pufmrg, state, sumvars, stubvar):
    df = pufmrg.copy()
    df.update(df.loc[:, sumvars].multiply(df[state], axis=0))
    state_sums = df.groupby(stubvar)[sumvars].sum().reset_index()
    grand_sums = state_sums[sumvars].sum().to_frame().transpose()
    grand_sums[stubvar] = 0
    state_sums = state_sums.append(grand_sums, ignore_index=True)
    state_sums[stubvar] = state_sums[stubvar].fillna(0)
    cols = state_sums.columns.tolist()
    state_sums['stgroup'] = state
    state_sums = state_sums[['stgroup'] + cols]
    state_sums.sort_values(by=stubvar, axis=0, inplace=True)
    # state_sums = state_sums.set_index(stubvar, drop=False)
    return state_sums


def get_allstates_wsums(pufsub, sweights):
    idvars = ['pid', 'filer', 'common_stub', 'ht2_stub']
    sumvars = [s for s in pufsub.columns if s not in idvars]
    stubvar  = 'ht2_stub'

    print('merging with weights...')
    pufmrg = pd.merge(pufsub,
                  sweights.drop(columns=['weight', 'geoweight_sum']),
                  how='left', on=['pid', 'ht2_stub'])

    print('looping through state groups to get weighted sums...')
    # loop through the state groups and get a summary dataframe for each
    state_groups = [s for s in sweights.columns if s not in ['pid', 'ht2_stub', 'weight', 'geoweight_sum']]
    dflist = []
    for stgroup in state_groups:
        print('  ' + stgroup)
        df = get_onestate_wsums(pufmrg, stgroup, sumvars, stubvar)
        dflist.append(df)

    allstates = pd.concat(dflist)
    allstates_long = pd.melt(allstates,
                             id_vars=['stgroup', 'ht2_stub'],
                             value_vars=sumvars,
                             var_name='pufvar', value_name='puf')
    allstates_long = pd.merge(allstates_long,
                              pc.irspuf_target_map.loc[:, ['pufvar', 'column_description']],
                              how='left', on='pufvar')

    return allstates_long


def get_compfile(allstates_long, ht2_compare):
    ht2keep = ['stgroup', 'pufvar', 'ht2var', 'target', 'ht2description', 'ht2_stub']
    comp = pd.merge(allstates_long,
                    ht2_compare.loc[:, ht2keep],
                    how='inner', on=['stgroup', 'pufvar', 'ht2_stub'])
    comp['diff'] = comp.puf - comp.target
    comp['pdiff'] = comp['diff'] / comp.target * 100
    comp['abspdiff'] = np.abs(comp.pdiff)

    # comp = pd.merge(comp, pc.irspuf_target_map, how='left', on='pufvar')
    comp = pd.merge(comp,
                    pc.ht2stubs.rename(columns={'ht2stub': 'ht2_stub'}),
                    how='left', on='ht2_stub')

    ordered_vars = ['stgroup', 'ht2_stub', 'ht2range', 'pufvar', 'ht2var',
                    'target', 'puf', 'diff', 'pdiff', 'abspdiff',
                    'ht2description']

    comp = comp[ordered_vars]

    comp.abspdiff.sort_values(ascending=False)
    return comp

