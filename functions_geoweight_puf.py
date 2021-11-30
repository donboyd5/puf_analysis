# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:58:25 2020

@author: donbo
"""

# %% imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import dask
# import dask.dataframe as dd
from timeit import default_timer as timer
import pickle

from collections import namedtuple

import puf_utilities as pu

# microweight - apparently we have to tell python where to find this
WEIGHTING_DIR = str(Path.home() / 'Documents/python_projects/weighting')
if WEIGHTING_DIR not in sys.path:
    sys.path.append(str(WEIGHTING_DIR))
import src.microweight as mw
import src.utilities as ut


# %% functions
def collapse_ht2(ht2_path, compstates):
    ht2_shares = pd.read_csv(ht2_path)

    # collapse target shares to these states and all others
    m_states = ht2_shares.state.isin(compstates)
    ht2_shares['stgroup'] = ht2_shares.state
    ht2_shares.loc[~m_states, 'stgroup'] = 'other'
    aggvars = ['stgroup', 'pufvar', 'ht2var', 'ht2description', 'ht2_stub']
    ht2_collapsed = ht2_shares.groupby(aggvars).agg({'share': 'sum', 'ht2': 'sum'}).reset_index()

    # put the sums over all state groups, for an income stub and pufvar group, on the file
    ht2sums = ht2_collapsed.groupby(['ht2_stub', 'pufvar'])[['ht2', 'share']].sum().rename(columns={'ht2': 'ht2sum', 'share': 'sharesum'}).reset_index()
    ht2_collapsed = pd.merge(ht2_collapsed, ht2sums, how='left', on=['ht2_stub', 'pufvar'])

    return ht2_collapsed


def get_geo_weights(
    df,
    weightdf,
    targvars,
    ht2wide,
    dropsdf_wide,
    independent,
    geomethod,
    options,
    intermediate_path=None):

    stub = df.name # DON'T EVEN PRINT df.name or all gets messed up
    print(f'\nIncome stub {stub:3d}')
    qx = '(ht2_stub == @stub)'

    # create local copy of weights with proper names
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])
    weightdf = weightdf.loc[:, ['pid', 'weight']]

    df = df.drop(columns='weight', errors='ignore')
    df = pd.merge(df, weightdf, how='left', on='pid')

    pufstub = df[['pid', 'ht2_stub', 'weight'] + targvars]

    wh = pufstub.weight.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()

    dropsdf_stub = dropsdf_wide.query(qx)[['stgroup'] + targvars]
    drops = np.asarray(dropsdf_stub[targvars], dtype=bool)  # True means we drop

    stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)

    # call the solver
    options_defaults = {'drops': drops, 'independent': independent, 'qmax_iter': 20}
    options_all = options_defaults.copy()
    options_all.update(options)

    # create Q_init matrix
    # if 'Q' in options_all and options_all['Q'] is not None:
    #     # create matrix from passed-in dataframe
    #     qshares = options_all['Q']
    #     qshares = qshares.query(qx).drop(columns=['pid', 'ht2_stub'])
    #     Q_init = qshares.to_numpy()
    # elif 'Q' not in options_all:
    #     # create initial Q, is n x m (# of households) x (# of areas)
    #     init_shares = (targetsdf.nret_all / targetsdf.nret_all.sum()).to_numpy()
    #     Q_init = np.tile(init_shares, (wh.size, 1))

    if 'qshares' in options_all and options_all['qshares'] is not None:
        print('qshares found and is not None')
        # create matrix from passed-in dataframe
        qshares = options_all['qshares']
        qshares = qshares.query(qx).drop(columns=['pid', 'ht2_stub'])
        Q_init = qshares.to_numpy()
    else:
        print('qshares not found or is found but None')
        # create initial Q, is n x m (# of households) x (# of areas)
        init_shares = (targetsdf.nret_all / targetsdf.nret_all.sum()).to_numpy()
        Q_init = np.tile(init_shares, (wh.size, 1))

    # print(Q_init.shape)
    options_all['Q'] = Q_init

    gw = stub_prob.geoweight(method=geomethod, options=options_all)
    # gw = stub_prob.geoweight(method='poisson', user_options=uo)
    whsdf = pd.DataFrame(gw.whs_opt, columns=sts)
    whsdf['geoweight_sum'] = whsdf.sum(axis=1)
    whsdf = whsdf[['geoweight_sum'] + sts]
    df2 = pd.concat([pufstub[['pid', 'ht2_stub', 'weight']],
                      whsdf],
                    axis=1)
    if intermediate_path is not None:
        df2.to_csv(intermediate_path + 'stub_' + str(stub) + '.csv', index=False)
    return df2


def get_geoweight_sums(pufsub,
                       weightdf,
                       targvars,
                       ht2wide,
                       dropsdf_wide,
                       outpath,
                       stubs=None):
    a = timer()
    geomethod = 'qmatrix-ipopt'
    options = {'quiet': True,
            # xlb, xub: lower and upper bounds on ratio of new state weights to initial state weights
            'xlb': 0.1,
            'xub': 100,
            'crange': .0001, # .0001 means within 0.01% of the target
            'linear_solver': 'ma57'
            }

    options = {'quiet': True,
            # xlb, xub: lower and upper bounds on ratio of new state weights to initial state weights
            'xlb': 0.1,
            'xub': 100,
            'crange': .0001, # .0001 means within 0.01% of the target
            'linear_solver': 'ma77',
            'output_file': '/home/donboyd5/Documents/test.out'
            }
    #'output_file': '/home/donboyd5/Documents/gwpi2.out',
    # 'print_user_options': 'yes',
    # 'file_print_level': 5,

    if stubs is None:
        grouped = pufsub.groupby('ht2_stub')
    else:
        grouped = pufsub.loc[pufsub.ht2_stub.isin(stubs)].groupby('ht2_stub')

    print("Starting loop through stubs...")
    weights_geosums = grouped.apply(get_geo_weights,
                                    weightdf=weightdf,
                                    targvars=targvars,
                                    ht2wide=ht2wide,
                                    dropsdf_wide=dropsdf_wide,
                                    independent=True,
                                    geomethod=geomethod,
                                    options=options)
    print("Done with loop through stubs.")
    print("Saving all geoweights (sums and state weights)...")
    weights_geosums = weights_geosums.droplevel(0).reset_index().drop(columns='index')
    weights_geosums = weights_geosums.sort_values(by='pid')
    weights_geosums.to_csv(outpath, index=False)
    b = timer()
    print("\nElapsed seconds: ", b - a)
    return weights_geosums.loc[:, ['pid', 'geoweight_sum']].rename(columns={'geoweight_sum': 'weight'})


# use qmatrix-ipopt because it seems most robust and is pretty fast
# qmatrix-lsq does not work as robustly as qmatrix-ipopt although it can be faster
# geomethod = 'qmatrix-lsq'
# options = {'verbose': 0,
#            'xlb': 0.2,
#            'scaling': False,
#            'method': 'bvls',  # bvls (default) or trf - bvls usually faster, better
#            'lsmr_tol': 'auto'  # 'auto'  # 'auto' or None
#            }


def get_geoweight_sums_direct(pufsub,
                       weightdf,
                       targvars,
                       ht2wide,
                       dropsdf_wide,
                       outpath,
                       stubs='all'):
    a = timer()

    if stubs == 'all':
        grouped = pufsub.groupby('ht2_stub')
    else:
        grouped = pufsub.loc[pufsub.ht2_stub.isin(stubs)].groupby('ht2_stub')

    print("Starting loop through stubs...")
    weights_geosums = grouped.apply(get_geo_weights_direct,
                                    weightdf=weightdf,
                                    targvars=targvars,
                                    ht2wide=ht2wide,
                                    dropsdf_wide=dropsdf_wide)
    print("Done with loop through stubs.")
    print("Saving all geoweights (sums and state weights)...")
    weights_geosums = weights_geosums.droplevel(0).reset_index()
    weights_geosums.to_csv(outpath, index=None)
    b = timer()
    print("Elapsed seconds: ", b - a)
    return weights_geosums.loc[:, ['pid', 'geoweight_sum']].rename(columns={'geoweight_sum': 'weight'})



def get_geo_weights_direct(
    df,
    weightdf,
    targvars,
    ht2wide,
    dropsdf_wide):

    print(f'\nIncome stub {df.name:3d}')
    stub = df.name
    qx = '(ht2_stub == @stub)'

    options = {
        'xlb': .1, 'xub': 10.,  # default 0.1, 10.0
        'crange': 0.0,  # default 0.0
        # 'print_level': 0,
        'file_print_level': 5,
        # 'scaling': True,
        # 'scale_goal': 1e3,
        'ccgoal': 10,
        'addup': False,  # default is false
        'output_file': '/home/donboyd/Documents/test_sparse.out',
        'max_iter': 100,
        'linear_solver': 'ma57',  # ma27, ma77, ma57, ma86 work, not ma97
        'quiet': False}

    # create local copy of weights with proper names
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])
    weightdf = weightdf.loc[:, ['pid', 'weight']]

    df['ht2_stub'] = df.name
    df = df.drop(columns='weight', errors='ignore')
    df = pd.merge(df, weightdf, how='left', on='pid')

    pufstub = df[['pid', 'ht2_stub', 'weight'] + targvars]

    wh = pufstub.weight.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()

    dropsdf_stub = dropsdf_wide.query(qx)[['stgroup'] + targvars]
    drops = np.asarray(dropsdf_stub[targvars], dtype=bool)  # True means we drop

    stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)

    # call the solver
    gw = stub_prob.geoweight(method='direct_ipopt', options=options)
    print("sspd: ", gw.sspd)

    whsdf = pd.DataFrame(gw.whs_opt, columns=sts)
    whsdf['geoweight_sum'] = whsdf.sum(axis=1)
    whsdf = whsdf[['geoweight_sum'] + sts]
    df2 = pd.concat([pufstub[['pid', 'ht2_stub', 'weight']],
                      whsdf],
                    axis=1)
    return df2


# %% one stub
def get_geo_weights_stub(
    df,
    weightdf,
    targvars,
    ht2wide,
    dropsdf_wide,
    method,
    options,
    stub,
    outdir,
    write_logfile):

    a = timer()

    if write_logfile:
        logpath = outdir + 'stub' + str(stub).zfill(2) + '_log.txt'
        f = open(logpath, 'w', buffering=1)
        original_stdout = sys.stdout # Save a reference to the original standard output
    else:
        f = sys.stdout

    print(f'\nIncome stub {stub:3d}', file=f)
    df = df.loc[df['ht2_stub']==stub]
    sub = ht2wide.loc[ht2wide.ht2_stub==stub, :]

    numrecs = df.shape[0]
    numgeo = sub.shape[0]
    numtargs = len(targvars)
    tottargs = numtargs * numgeo


    print(f'\nSolving geo weights for:', file=f)
    print(f'  {numrecs:10,d} records', file=f)
    print(f'  {numgeo:10,d} geographic areas', file=f)
    print(f'  {numtargs:10,d} targets per area', file=f)
    print(f'  {tottargs:10,d} targets in total', file=f)

    good = sub.loc[:, (sub.sum(axis=0) != 0)]

    if good.shape[1] < sub.shape[1]:
        dropcols = [var for var in targvars if not var in good.columns]
        numdrops = len(dropcols)
        totdrops = numdrops * numgeo
        print(f'\nWARNING: dropping {numdrops:d} targets where ht2 sum over all {numgeo:d} areas is ZERO:\n', dropcols, file=f)
        print(f'Total number dropped is {totdrops:,d}, leaving {tottargs - totdrops:,d}.', file=f)
        targvars = [var for var in targvars if var in good.columns]

    qx = '(ht2_stub == @stub)'

    # create local copy of weights with proper names
    weightdf = pu.idx_rename(weightdf, col_indexes=[0, 1], new_names=['pid', 'weight'])
    weightdf = weightdf.loc[:, ['pid', 'weight']]

    df = df.drop(columns='weight', errors='ignore')
    df = pd.merge(df, weightdf, how='left', on='pid')

    pufstub = df[['pid', 'ht2_stub', 'weight'] + targvars]

    wh = pufstub.weight.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()

    # once we have a numpy array we can fix zeros
    nzvalues = np.count_nonzero(targets)
    zvalues = targets.size - nzvalues
    if nzvalues < targets.size:
        print(f"\nWARNING: {zvalues:3d} of {targets.size:3d} targets are ZERO!", file=f)
        print('Replacing zeros with targets calculated using per-return value', file=f)
        print('for geo area with smallest nonzero per-return value target...', file=f)
        # https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
        # this relies on column zero having the number of returns for each state
        # we compute the per-return value for each target, by state
        # and find the smallest nonzero per-return value for each target
        # we then assign that smallest nonzero value to the zero-valued items
        state_avgs = np.divide(targets, targets[:,0].reshape((-1, 1)))
        smallest_nz_stateavg = np.ma.masked_equal(state_avgs, 0.0, copy=False).min(axis=0)
        # find indices of values we need to replace
        inds = np.where(targets==0)
        # place smallest values in locations defined by the indices; align the arrays using take
        state_avgs[inds] = np.take(smallest_nz_stateavg, inds[1])
        targets = state_avgs * targets[:,0].reshape((-1, 1))

    dropsdf_stub = dropsdf_wide.query(qx)[['stgroup'] + targvars]
    drops = np.asarray(dropsdf_stub[targvars], dtype=bool)  # True means we drop

    stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)

    # call the solver
    gw = stub_prob.geoweight(method=method, options=options, logfile=f)

    whsdf = pd.DataFrame(gw.whs_opt, columns=sts)
    whsdf['geoweight_sum'] = whsdf.sum(axis=1)
    whsdf = whsdf[['geoweight_sum'] + sts]
    # put stubs on the file
    whsdf = pd.concat([pufstub[['pid', 'ht2_stub', 'weight']],
                      whsdf],
                    axis=1)

    b = timer()

    # print summary info to stdout no matter what
    print(f'stub: {stub: 4d};  l2norm: {np.sqrt(gw.sspd): 11.3f};  seconds: {b - a: 8.2f}')

    # create a named tuple of items to return
    fields = ('elapsed_seconds',
              'whsdf',
              # 'geotargets_opt',
              'beta_opt')
    Result = namedtuple('Result', fields, defaults=(None,) * len(fields))

    result = Result(elapsed_seconds=b - a,
                    whsdf=whsdf,
                    # geotargets_opt=geotargets_opt,
                    beta_opt=gw.method_result.beta_opt)

    if write_logfile:
        # restore stdout and clean up
        sys.stdout = original_stdout
        f.close()

    return result


def callstub(stub, df, weightdf, targvars, ht2wide, dropsdf_wide,
             method, options, outdir, write_logfile):
    a = timer()
    gw = get_geo_weights_stub(
        df,
        weightdf,
        targvars,
        ht2wide,
        dropsdf_wide,
        method,
        options,
        stub,  # stub=None tells get_geo_weights_stub to get the stub number from df.name
        outdir,
        write_logfile)
    gw.whsdf.to_csv(outdir + 'stub' + str(stub).zfill(2) + '_whs.csv', index=False)
    # np.save(outdir + 'stub' + str(stub).zfill(2) + '_betaopt.npy', gw.beta_opt)

    open_file = open(outdir + 'stub' + str(stub).zfill(2) + '_betaopt.pkl', "wb")
    pickle.dump(gw.beta_opt, open_file)  # maybe find a better format?
    open_file.close()

    b = timer()
    return  b - a


def runstubs(
    stubs,
    pufsub,
    weightdf,
    targvars,
    ht2wide,
    dropsdf_wide,
    approach,
    options,
    outdir,
    write_logfile,
    parallel=False):
    if parallel:
        func = dask.delayed(callstub)
    else:
        func = callstub

    a = timer()
    output = []
    for stub in stubs:
        res = func\
            (stub,
            pufsub,
            weightdf,
            targvars,
            ht2wide,
            dropsdf_wide,
            approach,
            options,
            outdir,
            write_logfile)
        output.append(res)
    if parallel:
        total = dask.delayed()(output)
        total.compute(scheduler='threads')

    b = timer()

    elapsed_minutes = (b - a    ) / 60
    print('Elapsed minutes: ', np.round(elapsed_minutes, 2))
    return

