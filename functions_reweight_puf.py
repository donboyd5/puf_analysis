
import numpy as np
import pandas as pd
import sys
from functools import reduce

import puf_constants as pc
import puf_utilities as pu

# microweight - apparently we have to tell python where to find this
sys.path.append('c:/programs_python/weighting/')  # needed
import src.microweight as mw

def ulist(thelist):
    # return unique list with out changing order of original list
    ulist = []
    for x in thelist:
        if x not in ulist:
            ulist.append(x)
    return ulist

def get_possible_targets(targets_fname):
    targets_possible = pd.read_csv(targets_fname)

    target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
    target_vars = target_mappings.pufvar.to_list()

    # get names of puf variables for which we will need to create nnz indicator
    innz = target_mappings.pufvar.str.contains('_nnz')
    nnz_vars = target_mappings.pufvar[innz]
    pufvars_to_nnz = nnz_vars.str.rsplit(pat='_', n=1, expand=True)[0].to_list()

    possible_wide = targets_possible.loc[:, ['common_stub', 'pufvar', 'irs']] \
        .pivot(index='common_stub', columns='pufvar', values='irs') \
        .reset_index()
    possible_wide.columns.name = None
    return possible_wide


def merge_weights(weight_list, dir):
    wtpaths = [dir + s + '.csv' for s in weight_list]
    dflist = [pd.read_csv(file) for file in wtpaths]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['pid'],
                                                    how='outer'), dflist)
    return df_merged


def prep_puf(puf, targets):
    puf = puf.copy()

    target_names = targets.columns.tolist()
    target_names.remove('common_stub')

    # get unique list of variables and check if they are all in the puf
    vars = [s.replace('pos', '').replace('neg', '').replace('_nnz', '') for s in target_names]
    vars = ulist(vars)

    # create lists of variables to be created, without suffixes
    pos = [s.replace('pos', '') for s in target_names if ('pos' in s) and ('nnz' not in s)]
    neg = [s.replace('neg', '') for s in target_names if ('neg' in s) and ('nnz' not in s)]
    nnz = [s.replace('_nnz', '') for s in target_names if 'nnz' in s]

    puf['common_stub'] = pd.cut(
        puf['c00100'],
        pc.COMMON_STUBS,
        labels=range(1, 19),
        right=False)

    puf['ht2_stub'] = pd.cut(
        puf['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)

    puf['filer'] = pu.filers(puf)

    puf['nret_all'] = 1

    # marital status indicators
    puf['mars1'] = puf.MARS.eq(1)
    puf['mars2'] = puf.MARS.eq(2)
    puf['mars3'] = puf.MARS.eq(3)
    puf['mars4'] = puf.MARS.eq(4)
    puf['mars5'] = puf.MARS.eq(5)

    for var in pos:
        puf[var + 'pos'] = puf[var] * puf[var].gt(0)

    for var in neg:
        puf[var + 'neg'] = puf[var] * puf[var].lt(0)

    for var in nnz:
        puf[var + '_nnz'] = puf[var].ne(0) * 1

    # safely drop columns we don't want to keep
    idvars = ['pid', 'filer', 'common_stub']
    numvars = ['nret_all', 'mars1', 'mars2', 'mars3', 'mars4']
    keep_vars = idvars + numvars + target_names
    keep_vars = ulist(keep_vars)  # keeps unique names in case there is overlap with idvars

    return puf.loc[puf['filer'], keep_vars]


def puf_reweight(pufsub, targets, method='lsq'):
    grouped = pufsub.groupby('common_stub')
    new_weights = grouped.apply(stub_opt, targets, method=method)  # method lsq or ipopt
    return new_weights


# targets = ptargets.copy()
# method = 'lsq'
#
# stub = 3
# df = pufsub.query('common_stub == @stub').copy()
#
# rw.pdiff


def stub_opt(df, targets, method):
    # function to reweight a single stub of the puf
    print(df.name)
    stub = df.name

    target_names = targets.columns.tolist()
    target_names.remove('common_stub')

    # check initial percentage differences before removing any targets
    wh = np.asarray(df.weight)
    xmat = np.asarray(df[target_names], dtype=float)
    targvals = targets.loc[[stub], target_names]
    targets_stub = np.asarray(targvals, dtype=float).flatten()
    targets_calc = np.dot(xmat.T, wh)
    init_pdiff = targets_calc / targets_stub * 100 - 100
    idx_bad_pdiff = np.argwhere(np.abs(init_pdiff) > 150)

    # create a list of target names to use
    # first, remove the bad percentage differences
    target_names_array = np.array(target_names)
    targets_use = list(np.delete(target_names_array, idx_bad_pdiff))

    # always remove social security total
    badvars = ['e02400', 'e02400_nnz']
    targets_use = [pufvar for pufvar in targets_use if pufvar not in badvars]

    # more bad vars
    # badvars = ['e26270pos', 'e26270pos_nnz', 'e26270neg', 'e26270neg_nnz', 'c01000neg', 'c01000neg_nnz']
    # good badvars = ['e26270pos', 'e26270pos_nnz', 'e26270neg', 'e26270neg_nnz', 'c01000neg', 'c01000neg_nnz']
    badvars = ['e26270pos', 'e26270pos_nnz', 'e26270neg', 'e26270neg_nnz']
    targets_use = [pufvar for pufvar in targets_use if pufvar not in badvars]

    # now remove any stub-specific anticipated bad variables
    if stub in range(1, 5):
        # medical capped, and contributions
        badvars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']
        targets_use = [pufvar for pufvar in targets_use if pufvar not in badvars]

    # len(targets_use)

    # targets_use = target_names[0:24]
    # targets_use = targets_use[0:27]
    # targets_use[23]
    df2 = df[['pid', 'weight'] + targets_use].copy()
    targvals = targets.loc[[stub], targets_use]
    xmat = np.asarray(df2[targets_use], dtype=float)
    targets_stub = np.asarray(targvals, dtype=float).flatten()

    prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)
    # prob.pdiff_init

    if method == 'lsq':
        opts = {'xlb': 0.1, 'xub': 100, 'tol': 1e-7, 'method': 'bvls',
                'scaling': True,
                'max_iter': 50}  # bvls or trf
        # opts = {'xlb': 0.001, 'xub': 1000, 'tol': 1e-7, 'method': 'trf', 'max_iter': 500}
    elif method == 'ipopt':
        # opts = {'crange': 0.001, 'quiet': False}
        opts = {'crange': 0.001, 'xlb': 0.1, 'xub': 100, 'quiet': False}

    rw = prob.reweight(method=method, options=opts)
    # np.quantile(rw.g, qtiles)
    # rw.pdiff

    df2['reweight'] = df2.weight * rw.g
    return df2[['pid', 'weight', 'reweight']]


# prepare comp file and target_mappings
def pufsums(pufcomp):
    # prepare puf sums
    idvars = ['pid', 'common_stub', 'weight', 'weight_init']
    puflong = pufcomp.drop(columns='filer').melt(id_vars=idvars, var_name='pufvar')
    puflong['init'] = puflong.weight_init * puflong.value
    puflong['puf'] = puflong.weight * puflong.value
    pufsums = puflong.groupby(['common_stub', 'pufvar'])[['puf', 'init']].sum().reset_index()

    grand_sums = pufsums.groupby(['pufvar']).sum().reset_index()
    grand_sums['common_stub'] = 0
    pufsums = pufsums.append(grand_sums)
    return pufsums


# comparison report
def comp_report(pufsub, weights, weights_init, targets, outfile, title):
    pufcomp = pufsub.copy().drop(columns='weight')
    pufcomp = pd.merge(pufcomp, weights[['pid', 'weight']], how='inner', on='pid')
    pufcomp = pd.merge(pufcomp,
                       weights_init[['pid', 'weight']].rename(columns={'weight': 'weight_init'}),
                       how='inner', on='pid')

    print(f'Summarizing the puf...')
    psums = pufsums(pufcomp)

    print(f'Preparing the comparison file...')
    targets_long = pd.melt(targets, id_vars='common_stub', var_name='pufvar', value_name='irs')
    comp = pd.merge(targets_long, psums, how='inner', on=['common_stub', 'pufvar'])

    comp['diff'] = comp['puf'] - comp['irs']
    comp['pdiff'] = comp['diff'] / comp['irs'] * 100
    comp['ipdiff'] = comp['init'] / comp['irs'] * 100 - 100

    comp = pd.merge(comp, pc.irspuf_target_map, how='inner', on='pufvar')
    comp = pd.merge(comp, pc.irsstubs, how='inner', on='common_stub')

    # slim the file down
    ordered_vars = ['common_stub', 'incrange', 'pufvar', 'irsvar',
                    'irs', 'puf', 'diff', 'pdiff', 'ipdiff', 'column_description']
    comp = comp[ordered_vars]
    # sort by pufvar dictionary order (pd.Categorical)
    comp['pufvar'] = pd.Categorical(comp.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)

    comp.sort_values(by=['pufvar', 'common_stub'], axis=0, inplace=True)
    target_vars = comp.pufvar.unique()

    print(f'Writing report...')
    s = comp.copy()
    # define custom sort order
    # s['pufvar'] = pd.Categorical(s['pufvar'], categories=pufirs_fullmap.keys(), ordered=True)
    # s = s.sort_values(by=['pufvar', 'common_stub'])

    s['pdiff'] = s['pdiff'] / 100.0
    s['ipdiff'] = s['ipdiff'] / 100.0
    format_mapping = {'irs': '{:,.0f}',
                      'puf': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}',
                      'ipdiff': '{:.1%}'}
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
    tfile.write('\n\n2. Detailed report by AGI range for each variable:')
    for var in target_vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()

    return


