
import numpy as np
import pandas as pd
# from functools import reduce

import puf_constants as pc
import puf_utilities as pu

def comp_report(pdiff_df, outfile, title, ipdiff_df=None):

    # comparison report
    #   pdiff_df, and ipdiff_df if present, are data frames created by rwp.get_pctdiffs

    print(f'Preparing report...')
    comp = pdiff_df.copy()
    if ipdiff_df is not None:
        keep = ['common_stub', 'pufvar', 'pdiff']
        ipdiff_df.loc[:, keep].rename(columns={'pdiff': 'ipdiff'})
        comp = pd.merge(comp, ipdiff_df.loc[:, keep].rename(columns={'pdiff': 'ipdiff'}), how='left', on=['common_stub', 'pufvar'])

    # we already have column_description so don't bring it in again
    comp = pd.merge(comp, pc.irspuf_target_map.drop(columns=['column_description']), how='left', on='pufvar')  # get irsvar and other documentation


    ordered_vars = ['common_stub', 'incrange', 'pufvar', 'irsvar',
                    'target', 'puf', 'diff',
                    'pdiff', 'ipdiff', 'column_description']  # drop abspdiff
    if ipdiff_df is None: ordered_vars.remove('ipdiff')
    comp = comp[ordered_vars]

    # sort by pufvar dictionary order (pd.Categorical)
    comp['pufvar'] = pd.Categorical(comp.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)

    comp.sort_values(by=['pufvar', 'common_stub'], axis=0, inplace=True)

    target_vars = comp.pufvar.unique()

    print(f'Writing report...')
    s = comp.copy()

    s['pdiff'] = s['pdiff'] / 100.0
    if ipdiff_df is not None: s['ipdiff'] = s['ipdiff'] / 100.0
    format_mapping = {'target': '{:,.0f}',
                      'puf': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}',
                      'ipdiff': '{:.1%}'}
    if ipdiff_df is None: format_mapping.pop('ipdiff')
    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    tfile.write('Data are for filers only, using IRS filing requirements plus estimates of likely filing.\n')
    tfile.write("  Columns 'diff' and 'pdiff' give the puf value minus target, and diff as % of target, respectively.\n")
    tfile.write("  If column 'ipdiff' is present, it gives the % difference between puf and target using a different set of weights that should be mentioned in report title.\n")
    tfile.write('\nThis report is in 3 sections:\n')
    tfile.write('  1. Summary report for all variables, summarized over all filers\n')
    tfile.write('  2. Detailed report by AGI range for each variable\n')
    tfile.write('  3. Table that provides details on puf variables and their mappings to irs data\n')
    tfile.write('\n1. Summary report for all variables, summarized over all filers:\n\n')
    s2 = s[s.common_stub==0]
    tfile.write(s2.to_string())

    # now write details for each variable
    tfile.write('\n\n2. Detailed report by AGI range for each variable:')
    for var in target_vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    # finally, write the mapping
    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()
    print("All done.")

    return #  comp return nothing or return comp?



def ht2puf_report(ht2targets, outfile, title):

    # comparison report
    #   pdiff_df, and ipdiff_df if present, are data frames created by rwp.get_pctdiffs

    print(f'Preparing report...')
    # get list of variables in the pufvar dictionary order (pd.Categorical)
    df = ht2targets[['pufvar']].drop_duplicates()
    df['pufvar'] = pd.Categorical(df.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)
    pufvar_list = df.pufvar.tolist()

    # get the bad shares
    pufsums = ht2targets[['ht2_stub', 'pufvar', 'pufsum']].drop_duplicates()
    sharesums = ht2targets.groupby(['ht2_stub', 'pufvar', 'ht2var', 'column_description', 'ht2description'])[['ht2', 'share']].sum().reset_index()

    comp = pd.merge(sharesums, pufsums, how='left', on=['ht2_stub', 'pufvar'])
    comp = pd.merge(comp, pc.ht2stubs.rename(columns={'ht2stub': 'ht2_stub'}), how='left', on='ht2_stub') # bring in ht2range
    comp['diff'] = comp.ht2 - comp.pufsum
    comp['pdiff'] = comp['diff'] / comp.pufsum
    vorder = ['ht2_stub', 'ht2range',  'pufvar', 'ht2var', 'pufsum', 'ht2',
              'diff', 'pdiff', 'share',
              'column_description', 'ht2description']
    comp = comp[vorder]
    tol = 1e-4
    badshares = comp.query('share < (1 - @tol) or share > (1 + @tol)')

    print(f'Writing report...')
    s = comp.copy()
    s2 = badshares.copy()

    format_mapping = {
        'pufsum': '{:,.0f}',
        'ht2': '{:,.0f}',
        'diff': '{:,.0f}',
        'pdiff': '{:.1%}',
        'share': '{:.1%}'}

    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)
        s2[key] = s2[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    tfile.write('Comparison of weighted puf values and Historical Table 2 sums for the nation.\n')
    tfile.write('\nThis report is in 2 sections:\n')
    tfile.write('  1. Listing of ht2 variables and stubs for which shares do not add to 1. These may need to be dropped.\n')
    tfile.write('  2. Comparison, by variable, of weighted puf values and Historical Table 2 sums for the nation.\n')

    tfile.write('\nIn the tables that follow:\n')
    tfile.write('  - pufsum is the sum of puf values using our calculated weights.\n')
    tfile.write('  - ht2 is sum of state amounts reported in Historical Table 2 for included states.\n')
    tfile.write('  - diff is ht2 - pufsum\n')
    tfile.write('  - pdiff is diff as % of pufsum\n')
    tfile.write('  - share is the sum of the shares across the included states.\n')

    tfile.write('\n1. Bad shares: stub-variable combinations where state shares do not add to 100% (within small tolerance):\n\n')
    tfile.write(s2.to_string())

    # now write details for each variable
    tfile.write('\n\n2. Summary by AGI range for each variable:')
    for var in pufvar_list:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    # finally, write the mapping
    # tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    # tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()
    print("All done.")

    return
