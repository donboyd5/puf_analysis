# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:58:25 2020

@author: donbo
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer

import puf_utilities as pu

# microweight - apparently we have to tell python where to find this
weighting_dir = Path.home() / 'Documents/python_projects/weighting'
sys.path.append(str(weighting_dir))  # needed
import src.microweight as mw


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


def get_geo_weights(df, weightdf, targvars, ht2wide, dropsdf_wide,
                    independent,
                    geomethod,
                    options,
                    intermediate_path=None):

    print(f'\nIncome stub {df.name:3d}')
    stub = df.name
    qx = '(ht2_stub == @stub)'

    # create local copy of weights with proper names
    # weightdf.columns = ['pid', 'weight']
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
                       stubs='all'):
    a = timer()
    geomethod = 'qmatrix-ipopt'
    options = {'quiet': True,
            # xlb, xub: lower and upper bounds on ratio of new state weights to initial state weights
            'xlb': 0.1,
            'xub': 100,
            'crange': .0001, # .0001 means within 0.01% of the target
            'linear_solver': 'ma57'
            }


    if stubs == 'all':
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
    weights_geosums = weights_geosums.droplevel(0).reset_index()
    weights_geosums.to_csv(outpath, index=None)
    b = timer()
    print("Elapsed seconds: ", b - a)
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
