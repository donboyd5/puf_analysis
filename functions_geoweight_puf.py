# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:58:25 2020

@author: donbo
"""

import sys
import numpy as np
import pandas as pd

# microweight - apparently we have to tell python where to find this
sys.path.append('c:/programs_python/weighting/')  # needed
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

# stub = 1
# df =  pufsub.query('(ht2_stub == @stub)').copy()
# weightdf = weights.copy()

def get_georwt(df, weightdf, targvars, ht2wide, dropsdf_wide):
    print(df.name)
    stub = df.name

    weightdf.columns = ['pid', 'weight']  # force this df to have proper names

    df = df.copy().drop(columns='weight', errors='ignore')
    df = pd.merge(df, weightdf, how='left', on='pid')

    pufstub =df[['pid', 'ht2_stub', 'weight'] + targvars]

    wh = pufstub.weight.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    qx = '(ht2_stub == @stub)'
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()

    dropsdf_stub = dropsdf_wide.query(qx)[['stgroup'] + targvars]
    drops = np.asarray(dropsdf_stub[targvars], dtype=bool)  # True means we drop
    # print(targets.size)
    # print(drops.sum())

    # create initial Q, is n x m (# of households) x (# of areas)
    init_shares = (targetsdf.nret_all / targetsdf.nret_all.sum()).to_numpy()
    Q_init = np.tile(init_shares, (wh.size, 1))

    stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)

    # call the solver
    uo = {'Q': Q_init, 'drops': drops, 'independent': True}
    so = {'xlb': 0, 'xub': 50,
          'tol': 1e-7, 'method': 'bvls',
          'max_iter': 50, 'verbose': 0}
    gw_indep = stub_prob.geoweight(method='qmatrix-lsq', user_options=uo, solver_options=so)
    # gw_indep = stub_prob.geoweight(method='qmatrix-ec', user_options=uo)
    # gw_indep = stub_prob.geoweight(method='poisson', user_options=uo)
    df['geo_rwt'] = gw_indep.whs_opt.sum(axis=1)
    return df[['pid', 'ht2_stub', 'weight', 'geo_rwt']]


def get_geo_weights(df, weightdf, targvars, ht2wide, dropsdf_wide):
    print(df.name)
    stub = df.name

    weightdf.columns = ['pid', 'weight']  # force this df to have proper names

    df = df.copy().drop(columns='weight', errors='ignore')
    df = pd.merge(df, weightdf, how='left', on='pid')

    pufstub =df[['pid', 'ht2_stub', 'weight'] + targvars]

    wh = pufstub.weight.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    qx = '(ht2_stub == @stub)'
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()

    dropsdf_stub = dropsdf_wide.query(qx)[['stgroup'] + targvars]
    drops = np.asarray(dropsdf_stub[targvars], dtype=bool)  # True means we drop
    # print(targets.size)
    # print(drops.sum())

    # create initial Q, is n x m (# of households) x (# of areas)
    init_shares = (targetsdf.nret_all / targetsdf.nret_all.sum()).to_numpy()
    Q_init = np.tile(init_shares, (wh.size, 1))

    stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)

    # call the solver
    uo = {'Q': Q_init, 'drops': drops, 'independent': True}
    so = {'xlb': 0, 'xub': 50,
          'tol': 1e-7, 'method': 'bvls',
          'max_iter': 50, 'verbose': 0}
    gw = stub_prob.geoweight(method='qmatrix-lsq', user_options=uo, solver_options=so)
    # gw = stub_prob.geoweight(method='qmatrix-ec', user_options=uo)
    # gw = stub_prob.geoweight(method='poisson', user_options=uo)
    df['geo_rwt'] = gw.whs_opt
    return df[['pid', 'ht2_stub', 'weight', 'geo_rwt']]


