
import numpy as np
import pandas as pd
# from functools import reduce

import puf_constants as pc
import puf_utilities as pu

def wtdpuf_national_comp_report(pdiff_df, outfile, title, ipdiff_df=None):

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
    tfile.write(s2.to_string(index=False))

    # now write details for each variable
    tfile.write('\n\n2. Detailed report by AGI range for each variable:')
    for var in target_vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string(index=False))

    # finally, write the mapping
    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(pc.irspuf_target_map.to_string(index=False))
    tfile.close()
    print("All done.")

    return #  comp return nothing or return comp?


def ht2_vs_puf_report(ht2targets, outfile, title, outdir):

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

    outpath = outdir + 'ht2_vs_puf.csv'
    print('saving ht2 vs puf comparisons to: ', outpath)
    comp.to_csv(outpath, index=False)

    tol = 1e-4
    badshares = comp.query('share < (1 - @tol) or share > (1 + @tol)')

    # outpath = outdir + 'badshares.csv'
    # print(f'Writing badshares to ', outpath)
    # badshares.to_csv(outpath, index=False)

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
    tfile.write(s2.to_string(index=False))

    # now write details for each variable
    tfile.write('\n\n2. Summary by AGI range for each variable:')
    for var in pufvar_list:
        tfile.write('\n\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string(index=False))

    tfile.close()
    print("All done.")

    return


def ht2target_report(ht2targets, outfile, title, outpath):
    # determine which shares are far from a state's return share for a state, stub, pufvar

    def f(df):
        # function to be applied within stub-state-variable group to determine which shares are far from
        # shares for number of returns
        df['share_returns'] = df.share[df.pufvar=='nret_all'].values[0]
        df['share_diff'] = df.share - df.share_returns
        df['abs_diff'] = df['share_diff'].abs()
        return df

    print(f'Preparing report...')
    # get list of variables in the pufvar dictionary order (pd.Categorical)
    df = ht2targets[['pufvar']].drop_duplicates()
    df['pufvar'] = pd.Categorical(df.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)
    pufvar_list = df.pufvar.tolist()
    stgroup_list = ht2targets.stgroup.drop_duplicates().tolist()

    # determine the bad shares and exclude them
    badshares = ht2targets.groupby(['ht2_stub', 'pufvar'])[['share']].sum().reset_index().query('share < 0.999').drop(columns='share')
    badshares['bad'] = 1

    allshares = ht2targets.groupby(by=['stgroup', 'ht2_stub']).apply(f).reset_index()
    allshares = pd.merge(allshares, badshares, how='left', on=['ht2_stub', 'pufvar'])
    allshares = pd.merge(allshares, pc.ht2stubs.rename(columns={'ht2stub': 'ht2_stub'}), how='left', on='ht2_stub') # bring in ht2range
    print(f'Writing state share differences to ', outpath)
    allshares.to_csv(outpath, index=False)

    # prepare to write report
    comp = allshares[allshares.bad != 1].drop(columns='bad')
    comp['nshare_diff'] = comp.abs_diff  # keep a numeric version for sorting

    vorder = ['stgroup', 'ht2_stub', 'ht2range',  'pufvar', 'ht2var',
                'share_returns', 'share', 'share_diff',
                'column_description', 'nshare_diff']
    comp = comp[vorder]

    print(f'Writing report...')
    s = comp.copy()

    format_mapping = {
        'share_returns': '{:.1%}',
        'share': '{:12.1%}',
        'share_diff': '{:.1%}'}

    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    # tfile.write('Comparison of Historical Table 2 shares of the nation, by state, stub, and variable.\n')
    tfile.write('\nThis report is in 3 sections:\n')
    tfile.write('  1. Listing of the largest differences, across all variables and stubs, between return shares and shares for a specific variable.\n')
    tfile.write('  2. Listing of the largest differences, within each variable, between return shares and shares for a specific variable.\n')
    tfile.write('  3. Listing of the largest differences, within each state, between return shares and shares for a specific variable.\n')

    tfile.write('\nIn the tables that follow:\n')
    tfile.write('  - records are excluded for stub-variable combinations where national Historical Table 2 value is zero.\n')
    tfile.write('  - share_returns is the state\'s share of national returns for this stub.\n')
    tfile.write('  - share is the state\'s share of the nation for this variable in this stub.\n')
    tfile.write('  - share_diff is share - share_returns.\n')

    tfile.write('\n\n1. Largest share differences:\n\n')
    s2 = s.sort_values(by='nshare_diff', ascending=False).drop(columns='nshare_diff').head(50)
    tfile.write(s2.to_string(index=False))

    tfile.write('\n\n\n2. Largest share differences for each variable:')
    for var in pufvar_list:
        tfile.write('\n\n\n')
        s2 = s[s.pufvar==var].sort_values(by='nshare_diff', ascending=False).drop(columns='nshare_diff').head(15)
        tfile.write(s2.to_string(index=False))

    tfile.write('\n\n\n3. Largest share differences for each state:')
    for st in stgroup_list:
        tfile.write('\n\n\n')
        s2 = s[s.stgroup==st].sort_values(by='nshare_diff', ascending=False).drop(columns='nshare_diff').head(10)
        tfile.write(s2.to_string(index=False))

    tfile.close()
    print("All done.")

    return