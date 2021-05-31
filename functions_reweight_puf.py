
# %% imports
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from functools import reduce

import puf_constants as pc
import puf_utilities as pu

# microweight - we have to tell python where to find this
# sys.path.append('c:/programs_python/weighting/')  # needed
weighting_dir = Path.home() / 'Documents/python_projects/weighting'
# weighting_dir.exists()
sys.path.append(str(weighting_dir))  # needed

import src.microweight as mw

from timeit import default_timer as timer


# %% functions

def comp_report(pufsub,
                weights_reweight, weights_init,
                compvars, dropvars,
                outfile, title):

    # comparison report
    # weights_reweight.columns[[0, 1]] = ['pid', 'weight']  # force this df to have proper names
    # weights_init.columns[[0, 1]] = ['pid', 'weight']  # force this df to have proper names
    # create local copies of weights with proper names
    weights_reweight = pu.idx_rename(weights_reweight, col_indexes=[0, 1], new_names=['pid', 'weight'])
    weights_init = pu.idx_rename(weights_init, col_indexes=[0, 1], new_names=['pid', 'weight'])

    compvars_names = compvars.columns.tolist()
    compvars_names.remove('common_stub')

    print(f'Getting percent differences with initial weights...')
    keep = ['common_stub', 'pufvar', 'pdiff']
    ipdiffs = get_pctdiffs(pufsub, weights_init, compvars).loc[:, keep].rename(columns={'pdiff': 'ipdiff'})

    print(f'Getting percent differences with new weights...')
    # rwtsums = get_wtdsums(pufsub, target_names, weights_rwt)
    pdiffs = get_pctdiffs(pufsub, weights_reweight, compvars)

    print(f'Preparing report...')
    comp = pd.merge(pdiffs, ipdiffs, on=['common_stub', 'pufvar'])
    comp = pd.merge(comp, pc.irspuf_target_map, how='left', on='pufvar')
    comp = pd.merge(comp, pc.irsstubs, how='left', on='common_stub')

    ordered_vars = ['common_stub', 'incrange', 'pufvar', 'irsvar',
                    'target', 'puf', 'diff',
                    'pdiff', 'ipdiff', 'column_description']  # drop abspdiff
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
    s['ipdiff'] = s['ipdiff'] / 100.0
    format_mapping = {'target': '{:,.0f}',
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

    # finally, write the mapping
    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()

    return #  comp return nothing or return comp?


def get_pctdiffs(pufsub, weightdf, targets):
    # create a local copy of weight df with properly named columns
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])

    target_names = targets.columns.tolist()
    target_names.remove('common_stub')
    keepvars = ['common_stub', 'pid'] + target_names

    dfsums = get_wtdsums(pufsub.loc[:, keepvars], target_names, weightdf.iloc[:, [0, 1]])
    sumslong = pd.melt(dfsums, id_vars='common_stub', var_name='pufvar', value_name='puf')
    targetslong = pd.melt(targets, id_vars='common_stub', var_name='pufvar', value_name='target')
    dfmerge = pd.merge(sumslong, targetslong, on=['common_stub', 'pufvar'])
    dfmerge['diff'] = dfmerge.puf - dfmerge.target
    dfmerge['pdiff'] = dfmerge['diff'] / dfmerge.target * 100
    dfmerge['abspdiff'] = np.abs(dfmerge.pdiff)

    # bring in useful information about vars and stubs
    dfmerge = pd.merge(dfmerge, pc.irspuf_target_map[['pufvar', 'column_description']], how='left', on='pufvar')
    dfmerge = pd.merge(dfmerge, pc.irsstubs, how='left', on='common_stub')
    dfmerge = dfmerge.sort_values(by='abspdiff', ascending=False)

    return dfmerge


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


def get_wtdsums(pufsub, sumvars, weightdf, stubvar='common_stub'):
    # create local weights df with proper names
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])

    df = pufsub.copy().drop(columns='weight', errors='ignore')
    varnames = df.columns.tolist()
    varnames.remove(stubvar)
    varnames.remove('pid')

    df = pd.merge(df, weightdf.iloc[:, [0, 1]], how='left', on='pid')

    df.update(df.loc[:, sumvars].multiply(df.weight, axis=0))
    dfsums = df.groupby(stubvar)[sumvars].sum().reset_index()
    grand_sums = dfsums[sumvars].sum().to_frame().transpose()
    grand_sums[stubvar] = 0
    dfsums = dfsums.append(grand_sums, ignore_index=True)
    dfsums[stubvar] = dfsums[stubvar].fillna(0)
    dfsums.sort_values(by=stubvar, axis=0, inplace=True)
    dfsums = dfsums.set_index(stubvar, drop=False)
    return dfsums


def get_wtdsums_geo(pufsub, sumvars, weightdf, stubvar='common_stub'):
    # create local weights df with proper names
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])

    df = pufsub.copy().drop(columns='weight', errors='ignore')
    varnames = df.columns.tolist()
    varnames.remove(stubvar)
    varnames.remove('pid')

    df = pd.merge(df, weightdf.iloc[:, [0, 1]], how='left', on='pid')

    df.update(df.loc[:, sumvars].multiply(df.weight, axis=0))
    dfsums = df.groupby(stubvar)[sumvars].sum().reset_index()
    grand_sums = dfsums[sumvars].sum().to_frame().transpose()
    grand_sums[stubvar] = 0
    dfsums = dfsums.append(grand_sums, ignore_index=True)
    dfsums[stubvar] = dfsums[stubvar].fillna(0)
    dfsums.sort_values(by=stubvar, axis=0, inplace=True)
    dfsums = dfsums.set_index(stubvar, drop=False)
    return dfsums


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
    # avoid categorical variable, it causes problems!
    puf['common_stub'] = puf.common_stub.astype('int64')

    puf['ht2_stub'] = pd.cut(
        puf['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)
    # avoid categorical variable, it causes problems!
    puf['ht2_stub'] = puf.ht2_stub.astype('int64')

    puf['nret_all'] = 1

    # marital status indicators
    puf['mars1'] = puf.MARS.eq(1)
    puf['mars2'] = puf.MARS.eq(2)
    puf['mars3'] = puf.MARS.eq(3)
    puf['mars4'] = puf.MARS.eq(4)
    puf['mars5'] = puf.MARS.eq(5)

    # create any additional needed puf vars
    puf['taxac_irs'] = np.maximum(0, puf.c09200 - puf.niit - puf.refund)

    for var in pos:
        puf[var + 'pos'] = puf[var] * puf[var].gt(0)

    for var in neg:
        puf[var + 'neg'] = puf[var] * puf[var].lt(0)

    for var in nnz:
        puf[var + '_nnz'] = puf[var].ne(0) * 1

    # safely drop columns we don't want to keep
    idvars = ['pid', 'filer', 'common_stub', 'ht2_stub']
    numvars = ['nret_all', 'mars1', 'mars2', 'mars3', 'mars4']
    keep_vars = idvars + numvars + target_names
    keep_vars = ulist(keep_vars)  # keeps unique names in case there is overlap with idvars

    return puf.loc[puf['filer'], keep_vars]


def puf_reweight(pufsub, init_weights, targets, method='lsq', drops=None):

    a = timer()
    # create local copy of init_weights with columns pid, weight
    init_weights = pu.idx_rename(init_weights, col_indexes=[0, 1], new_names=['pid', 'weight'])

    pufsub = pufsub.copy()
    pufsub = pd.merge(pufsub.drop(columns='weight', errors='ignore'),
                      init_weights, on='pid', how='left')

    grouped = pufsub.groupby('common_stub')

    new_weights = grouped.apply(stub_opt, targets, method=method, drops=drops)  # method lsq or ipopt
    b = timer()
    print('Elapsed seconds: ', b - a)

    return new_weights


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



# targets = ptargets.copy()
# method = 'lsq'
#
# stub = 3
# df = pufsub.query('common_stub == @stub').copy()
#
# rw.pdiff


def stub_opt(df, targets, method, drops=None):
    # function to reweight a single stub of the puf
    print(f'\nIncome stub {df.name:3d}')
    stub = df.name

    target_names = targets.columns.tolist()
    target_names.remove('common_stub')

    drop_vars = []
    if drops is not None:
        drop_vars = drops[drops.common_stub==stub].pufvar.tolist()

    targets_use = [pufvar for pufvar in target_names if pufvar not in drop_vars]

    df = df[['pid', 'weight'] + targets_use]
    wh = np.asarray(df.weight)
    targvals = targets.loc[[stub], targets_use]
    xmat = np.asarray(df[targets_use], dtype=float)
    targets_stub = np.asarray(targvals, dtype=float).flatten()

    prob = mw.Microweight(wh=wh, xmat=xmat, targets=targets_stub)
    # prob.pdiff_init

    if method == 'lsq':
        opts = {'xlb': 0.1, 'xub': 100,
                'tol': 1e-7, 'method': 'bvls',
                'scaling': False,
                'max_iter': 50}  # bvls or trf
        # opts = {'xlb': 0.001, 'xub': 1000, 'tol': 1e-7, 'method': 'trf', 'max_iter': 500}
    elif method == 'ipopt' or method == 'ipopt_sparse':
        # opts = {'crange': 0.001, 'quiet': False}
        opts = {'crange': 0.001, 'xlb': 0.1, 'xub': 100, 'quiet': True, 'ccgoal': 1}

    # print(opts)
    rw = prob.reweight(method=method, options=opts)

    # print()  # get ipopt solver message -- via microweight
    qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)
    print('Quantiles for % differences from targets: ', qtiles)
    print(np.quantile(rw.pdiff, qtiles), '\n')

    new_weights = df[['pid', 'weight']].rename(columns={'weight': 'weight_init'})
    new_weights['weight'] = new_weights.weight_init * rw.g
    return new_weights[['pid', 'weight', 'weight_init']]


def ulist(thelist):
    # return unique list without changing order of original list
    ulist = []
    for x in thelist:
        if x not in ulist:
            ulist.append(x)
    return ulist

