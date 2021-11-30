
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


def ht2_vs_puf_report(ht2targets, outfile, title, outpath):

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

    # outpath = outdir + 'ht2_vs_puf.csv'
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


def calc_save_statesums(
    pufsub,
    state_weights,
    pufvars,
    outfile):

    idvars = ['pid', 'filer', 'ht2_stub']

    # get names of the states
    geos = state_weights.columns.tolist()
    nonstates = ['pid', 'ht2_stub', 'weight']  # keep geoweight_sum
    geos = [s for s in geos if s not in nonstates]

    # make a long puf file
    # BEFORE making the long file, convert boolean to integer
    # pufsub = pufsub * 1  # bool to integer
    # pufsub.replace({False: 0, True: 1}, inplace=True)
    # https://stackoverflow.com/questions/17383094/how-can-i-map-true-false-to-1-0-in-a-pandas-dataframe/27362540
    # pufsub.replace({False: 0, True: 1}, inplace=True)
    puflocal = pufsub.replace({False: 0, True: 1}).copy()
    puflong = puflocal.loc[:, idvars + pufvars] \
        .melt(id_vars=idvars, var_name='pufvar', value_name='pufvalue')

    puflong = pd.merge(puflong, state_weights.drop(columns='weight'), on=['pid', 'ht2_stub'], how='left')

    # multiply all geoweight_sum and state weight columns by the pufvalue column
    # this next line is the problem
    puflong.loc[:, geos] = puflong.loc[:, geos].mul(puflong.pufvalue, axis=0)

    # collapse by ht2_stub
    calcsums = puflong.groupby(['ht2_stub', 'pufvar'])[geos].sum().reset_index()
    calcsums.to_csv(outfile, index=False)

    return


def state_puf_vs_targets_report(
    state_targets,
    state_sums,
    title,
    reportfile):

    # get the calculated sums and add an hstub zero
    compsums = pd.read_csv(state_sums)
    # compsums = compsums.fillna(0)
    # set it up so that we easily get US totals, by renaming geoweight_sum to US
    compsums.rename(columns={"geoweight_sum": "US"}, inplace=True)
    idvars = ['ht2_stub', 'pufvar']
    compsums = compsums.melt(id_vars=idvars, var_name='stgroup', value_name='calcsum')
    comptotals = compsums.drop(columns='ht2_stub').groupby(['pufvar', 'stgroup']).sum().reset_index()
    # comptotals = comptotals.fillna(0)
    comptotals['ht2_stub'] = 0
    comptotals = pd.concat((compsums, comptotals), axis=0).sort_values(by=['ht2_stub', 'pufvar'])

    # Calculate US sums for the state targets file
    # 		pufvar		pufsum	ht2sum	share	sharesum	ht2	target
    groupvars = ['ht2_stub', 'pufvar', 'ht2var', 'column_description', 'ht2description']
    targtotals = state_targets.drop(columns='stgroup').groupby(groupvars).sum().reset_index()
    targtotals['stgroup'] = 'US'
    targtotals = pd.concat((state_targets, targtotals), axis=0).sort_values(by=['ht2_stub', 'pufvar'])


    mrgvars = ['ht2_stub', 'stgroup', 'pufvar']
    # targtotals = targtotals.fillna(0)
    comp = pd.merge(targtotals, comptotals, on=mrgvars, how='left')
    comp = pd.merge(comp, pc.ht2stubs.rename(columns={'ht2stub': 'ht2_stub'}), how='left', on='ht2_stub') # bring in ht2range
    # print(comp.loc[comp.stgroup=='US'])
    # comp = comp.fillna(0)  # CAUTION ???
    # print(comp.info())
    # print(comp.iloc[:10, :])

    comp['d_ht2'] = comp['calcsum'] - comp['ht2']
    comp['pd_ht2'] = comp['d_ht2'] / comp['ht2']

    comp['d_target'] = comp['calcsum'] - comp['target']
    comp['pd_target'] = comp['d_target'] / comp['target']
    comp['apd_target'] = comp['pd_target'].abs()  # keep a numeric version for sorting

    # 'ht2range'
    vorder = ['stgroup', 'ht2_stub', 'ht2range',
               'pufvar', 'ht2var',
               # 'pufsum', 'htwsum', # national??
               'ht2', 'target', 'calcsum',
               'd_ht2', 'pd_ht2',
               'd_target', 'pd_target',
               'column_description',
               'apd_target']
    comp = comp[vorder]
    # print(comp)

    print(f'Writing report...')
    pufvar_list = comp.pufvar.unique().tolist()
    stub_list = comp.ht2_stub.unique().tolist()
    state_list = comp.stgroup.unique().tolist()
    # comp = comp.sort_values(by=['apd_target'], ascending=False)

    s = comp.copy()

    format_mapping = {
        'ht2': '{:,.0f}',
        'target': '{:,.0f}',
        'calcsum': '{:,.0f}',
        'd_ht2': '{:,.0f}',
        'pd_ht2': '{:.1%}',
        'd_target': '{:,.0f}',
        'pd_target': '{:.1%}',}

    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(reportfile, 'a')
    tfile.truncate(0)

    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    # tfile.write('Comparison of Historical Table 2 shares of the nation, by state, stub, and variable.\n')

    tfile.write('\nThis report is in sections:\n')
    tfile.write('  1. Differences for US totals, by income stub, in pufvar order.\n')
    tfile.write('  2. 20 largest absolute % differences for each stub-pufvar combination.\n')
    tfile.write('  3. Differences within state, pufvar, stub.\n')

    # d_ht2 pd_ht2 d_target pd_target
    tfile.write('\nIn the tables, we have columns d_ht2, pd_ht2, d_target, and pd_target:\n')
    tfile.write('  The d_ prefix means the difference between a calculated amount and the value defined by the suffix.\n')
    tfile.write('  The pd_ prefix means the percentage difference between a calculated amount and the value defined by the suffix.\n')
    tfile.write('  The ht2 suffix means the comparison value is from Historical Table 2, as published.\n')
    tfile.write('  The target suffix means the comparison value is a target we developed based on totals computed from the puf.\n\n')

    tfile.write('  The ht2 and target values may differ for several reasons:\n')
    tfile.write('    For example, the HT2 variable may not defined the same as the puf variable is defined.\n')
    tfile.write('    Or the puf value may differe because the puf data simply are not consistent with the HT values.\n\n')
    tfile.write('  The ht2 comparisons tell us what other people might see if they compare to Historical Table 2 published amounts.\n')
    tfile.write('  The target comparisons tell us how well we did hitting targets that we think are most appropriate.\n')

    # U.S. differences
    tfile.write('\n\n1. Differences for US totals, by income stub, in pufvar order:\n')
    s = s.sort_values(by=['pufvar'])
    for stub in stub_list:
        tfile.write('\n\n')
        # s2 = s.loc[(s['pufvar'] == var) & (s['ht2_stub'] == stub) & (s['stgroup'] == 'US')].drop(columns='apd_target')
        s2 = s.loc[(s['ht2_stub'] == stub) & (s['stgroup'] == 'US')].drop(columns='apd_target')
        tfile.write(s2.to_string(index=False))

    # Largest differences
    tfile.write('\n\n2. 20 largest absolute % differences for each stub-pufvar combination:\n')
    s = s.sort_values(by=['apd_target'], ascending=False)
    for stub in stub_list:
        for var in pufvar_list:
            tfile.write('\n\n')
            s2 = s.loc[(s['pufvar'] == var) & (s['ht2_stub'] == stub)].iloc[:20, ].drop(columns='apd_target')
            # s2 = s[s.pufvar==var].iloc[:20, ].drop(columns='apd_target')
            tfile.write(s2.to_string(index=False))

    # Differences by state
    tfile.write('\n\n3. Differences within state, pufvar, stub:\n')
    s = s.sort_values(by=['stgroup', 'pufvar', 'ht2_stub'])
    for state in state_list:
        for var in pufvar_list:
            tfile.write('\n\n')
            s2 = s.loc[(s['stgroup'] == state) & (s['pufvar'] == var)].drop(columns='apd_target')
            tfile.write(s2.to_string(index=False))

    tfile.close()
    print("All done.")




    return

#  0   stgroup             15147 non-null  object
#  1   ht2_stub            15147 non-null  int64
#  2   pufvar              15147 non-null  object
#  3   ht2var              15147 non-null  object
#  4   pufsum              15147 non-null  float64
#  5   ht2sum              15147 non-null  float64
#  6   share               15147 non-null  float64
#  7   sharesum            15147 non-null  float64
#  8   ht2                 15147 non-null  float64
#  9   target              15147 non-null  float64
#  10  column_description  15147 non-null  object
#  11  ht2description      15147 non-null  object

# geoweight_sum
# calcsum