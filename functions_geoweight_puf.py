# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:58:25 2020

@author: donbo
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

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


    # df = df.copy().drop(columns='weight', errors='ignore')
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
        df2.to_csv(intermediate_path + 'stub_' + str(stub) + '.csv', index=None)
    return df2



