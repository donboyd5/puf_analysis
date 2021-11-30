# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 04:18:49 2020

@author: donbo
"""

# %% notes
# Caution: keep show variables (in python console) off until needed as it slows things down


# %% imports
import sys
import taxcalc as tc
import pandas as pd
import numpy as np

from timeit import default_timer as timer
from datetime import date

import puf_constants as pc
import puf_utilities as pu

# microweight - this is sufficient
WEIGHTING_DIR = str(Path.home() / 'Documents/python_projects/weighting')
if WEIGHTING_DIR not in sys.path:
    sys.path.append(str(WEIGHTING_DIR))
# sys.path.append('c:/programs_python/weighting/')  # needed
import src.microweight as mw

import src.make_test_problems as mtp

import src.reweight_leastsquares as rwls


# %% locations and file names
DATADIR = r'C:\programs_python\puf_analysis\data/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'

PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'
PUF_REGROWN_REWEIGHTED = PUFDIR + 'puf2017_regrown_reweighted.parquet'
PUF_REGROWN_RERE_REWEIGHTED = PUFDIR + 'puf2017_regrown_rere_reweighted.parquet'


# %% constants
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% functions
def uvals(series):
    return sorted(series.unique())

def flat(l):
    return ", ".join(l)


# %% get ht2_shares
ht2_shares = pd.read_csv(DATADIR + 'ht2_shares.csv')
ht2_shares
ht2_shares.info()
sts = uvals(ht2_shares.state)
len(sts)  # 52
flat(sts)  # includes DC, OA, but not PR or US


# %% define states to target
compstates = ('NY', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX', 'VT')

# collapse target shares to these states and all others
m_states = ht2_shares.state.isin(compstates)
ht2_shares['stgroup'] = ht2_shares.state
ht2_shares.loc[~m_states, 'stgroup'] = 'other'
uvals(ht2_shares.stgroup)
aggvars = ['stgroup', 'pufvar', 'ht2var', 'ht2description', 'ht2_stub']
ht2_collapsed = ht2_shares.groupby(aggvars).agg({'share': 'sum', 'ht2': 'sum'}).reset_index()
ht2_collapsed.info()


# %% get relevant national puf
puf = pd.read_parquet(PUF_REGROWN_REWEIGHTED)
puf


# %% add puf variables, get puf sums

potential_targets = uvals(ht2_collapsed.pufvar)
flat(potential_targets)

pvtonnz = ['c02500', 'c17000', 'c18300', 'c19700',
           'e00200', 'e00300', 'e00600']
puf = pu.prep_puf(puf, pufvars_to_nnz=pvtonnz)
# puf2 is puf # true!
puf.info()
flat(sorted(puf.columns))

keepvars = ['ht2_stub', 'pid', 's006_rwt'] + potential_targets
pufsub = puf.loc[puf.filer, keepvars]
puflong = pufsub.melt(id_vars=['ht2_stub', 's006_rwt'], var_name='pufvar')
puflong['wvalue'] = puflong.value * puflong.s006_rwt
puflong.shape
puflong.info()
aggvars = ['ht2_stub', 'pufvar']
pufsums = puflong.groupby(aggvars).agg({'wvalue': 'sum'}).reset_index()


# %% link pufsums to ht2 shares and create targets
ht2_collapsed
ht2targets = pd.merge(ht2_collapsed, pufsums, on=['pufvar', 'ht2_stub'])
ht2targets['target'] = ht2targets.wvalue * ht2targets.share
ht2targets['diff'] = ht2targets.target - ht2targets.ht2
ht2targets['pdiff'] = ht2targets['diff'] / ht2targets.ht2 * 100
ht2targets['abspdiff'] = np.abs(ht2targets['pdiff'])

keepvars = ['stgroup', 'ht2_stub', 'pufvar', 'target']
ht2wide = ht2targets[keepvars].pivot(index=['stgroup', 'ht2_stub'], columns='pufvar', values='target').reset_index()


# %% get adjusted national weights based on independence
pufsub.columns
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])

targvars = ['nret_all', 'mars1', 'mars2', 'c00100', 'e00200', 'e00200_nnz',
            'e00300', 'e00300_nnz', 'e00600', 'e00600_nnz',
            # deductions
            'c17000','c17000_nnz',
            'c18300', 'c18300_nnz']

def get_rwt(df, targvars, ht2wide):
    print(df.name)
    stub = df.name
    qx = '(ht2_stub == @stub)'
    pufstub =df[['pid', 'ht2_stub', 's006_rwt'] + targvars]

    wh = pufstub.s006_rwt.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()
    drops = np.where(targets == 0, True, False)  # True means we drop
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
    df['s006_rwt_geo'] = gw_indep.whs_opt.sum(axis=1)
    return df


grouped = pufsub.groupby('ht2_stub')

a = timer()
pufsub_rwtd = grouped.apply(get_rwt, targvars, ht2wide)
b = timer()
b - a

# pufsub_rwtd['s006_rwt_rwt'] = pufsub_rwtd.s006 * pufsub_rwtd.x
pufsub_rwtd.info()

# make the new puf
weights = pufsub_rwtd[['pid', 's006_rwt', 's006_rwt_geo']]
weights.s006_rwt.sum()
weights.s006_rwt_geo.sum()

np.nanquantile(weights.s006_rwt_geo / weights.s006_rwt, qtiles)


# %% merge new weights back with original weights
pufrgrw = pd.read_parquet(PUF_REGROWN_REWEIGHTED)
pufrgrw = pd.merge(pufrgrw, weights[['pid', 's006_rwt_geo']], on=['pid'], how='left')
pufrgrw.columns
rwmask = ~pufrgrw.s006_rwt_geo.isna()
rwmask.sum()
pufrgrw.loc[rwmask, ['s006']] = pufrgrw.s006_rwt_geo[rwmask]

pufrgrw[['pid', 's006', 's006_taxcalc', 's006_rwt', 's006_rwt_geo']].tail(100)
pufrgrw[['pid', 's006', 's006_taxcalc', 's006_rwt', 's006_rwt_geo']].head(100)

pufrgrw['s006'] = np.where(pufrgrw.s006_rwt_geo.isna(),
                           pufrgrw.s006_taxcalc,
                           pufrgrw.s006_rwt_geo)
pufrgrw.s006.sum()
pufrgrw.s006_taxcalc.sum()
pufrgrw.s006_rwt.sum()
pufrgrw.s006_rwt_geo.sum()

pufrgrw.to_parquet(PUFDIR + 'puf2017_regrown_reweighted_geoweighted.parquet', engine='pyarrow')



# %% examine individual stubs

def get_stub(stub, targvars, ht2wide):
    qx = '(ht2_stub == @stub)'
    pufstub = pufsub.query(qx)[['pid', 'ht2_stub', 's006_rwt'] + targvars]

    wh = pufstub.s006_rwt.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)

    # set up targets - keep a dataframe and a matrix even though dataframe
    # is not absolutely necessary
    targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
    sts = targetsdf.stgroup.tolist()
    targets = targetsdf[targvars].to_numpy()
    drops = np.where(targets == 0, True, False)  # True means we drop
    print(targets.size)
    print(drops.sum())

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
    return gw_indep



stub = 10
pufstub = pufsub.query('(ht2_stub == @stub)')[['pid', 'ht2_stub', 's006_rwt'] + targvars]
wh = pufstub.s006_rwt.to_numpy()
wh = np.where(wh <= .01, .01, wh)
np.quantile(wh, qtiles)

stub = 1
tdf, gw = get_stub(stub, targvars, ht2wide)

gw.method_result.geotargets
dir(gw)
gw.pdiff
whs = gw.whs_opt
gw.whs_opt.shape
gw.whs_opt.sum(axis=0)
ratio = gw.whs_opt.sum(axis=1) / wh
np.quantile(ratio, qtiles)
wh
type(get_stub(stub, targvars, ht2wide))


# %% prepare a single stub for geoweighting
pufsub.columns
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])

idx = (0, 1, 2, 3)
idx = slice(2, 2 + 5)
targvars = potential_targets[idx]

idx = [2, 3]
idx = [0, 3]
targvars = [potential_targets[index] for index in idx]
targvars = ['nret_all', 'mars1', 'mars2', 'c00100', 'e00200', 'e00200_nnz',
            'e00300', 'e00300_nnz', 'e00600', 'e00600_nnz',
            # deductions
            'c17000','c17000_nnz',
            'c18300', 'c18300_nnz']

stub = 3
pufstub = pufsub.query('ht2_stub == @stub')[['pid', 'ht2_stub', 's006_rwt'] + targvars]
pufstub
pufstub.describe()

wh = pufstub.s006_rwt.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
xmat.shape

targetsdf = ht2wide.loc[ht2wide.ht2_stub == stub, ['stgroup'] + targvars]
# create initial Q, is n(h) x m(s)
# df.query('A < 0.5 and B < 0.5')
init_shares = ht2_collapsed.query('(ht2_stub==@stub) and (pufvar=="nret_all")')[['stgroup', 'share']]
init_shares = init_shares.share.to_numpy()
Q_init = np.tile(init_shares, (wh.size, 1))

targets = targetsdf[targvars].to_numpy()
stub_prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=targets)
uo = {'max_iter': 1, 'Q': Q_init}

gw1 = stub_prob.geoweight(method='qmatrix', user_options=uo)
g1 = gw1.method_result.Q_unadjusted.sum(axis=1)
np.quantile(g1, qtiles)

uo = {'max_iter': 1, 'Q': Q_init, 'independent': True}
opts = {'xlb': 0, 'xub': 50, 'tol': 1e-7, 'method': 'bvls', 'max_iter': 50}
gw1a = stub_prob.geoweight(method='qmatrix-lsq', user_options=uo, solver_options=opts)
gw1a.sspd
gw1a.pdiff
np.quantile(gw1a.pdiff, qtiles)


pdiff_df = pd.DataFrame(data = gw1a.pdiff,
                       index = ht2_collapsed.stgroup.unique(),
                       columns=targvars).reset_index()

pdiffl = pd.melt(pdiff_df, id_vars='index', value_name='pdiff').rename(columns={'index': 'stgroup'})
pdiffl['abspdiff'] = pdiffl.pdiff.abs()
pdiffl.query('stgroup=="NY"').sort_values(by=['abspdiff'], ascending=False)
pdiffl.sort_values(by=['abspdiff'], ascending=False)

# try to reproduce qmatrix approach using lsq reweight, just one state
xmat_wh = xmat * wh.reshape((-1, 1))

j = 5  # NY
targetsdf.iloc[j, :]
whj = Q_init[:, j]
targetsj = targets[j, :]
# rwls.rw_lsq(wh, xmat, targets, options=lsq_opts)
lsq_opts = {
    'xlb': 1e-8,
    'xub': 100.0,
    'verbose': 2
    }
res = rwls.rw_lsq(whj, xmat_wh, targetsj, options=lsq_opts)
dir(res)
res.targets_opt
targetsj
res.targets_opt / targetsj * 100 - 100
g = res.g
np.isnan(g).any() or np.isinf(g).any() or g.any() == 0
Q_init[:, j] = Q_init[:, j] * g.reshape(g.size, )

whs = np.multiply(Q_init, wh.reshape((-1, 1)))  # faster
esttargs = np.dot(whs.T, xmat)
esttargs[j, :]

diff = np.dot(whs.T, xmat) - geotargets



            # g = gfn(Q[:, j],
            #         xmat_wh[:, good_cols],
            #         geotargets[j, good_cols],
            #         objective=so.objective)


dir(gw1a.method_result)
g = gw1a.method_result.Q_unadjusted.sum(axis=1)
g.shape
g.sum()
g[range(10)]
np.quantile(g, qtiles)
new_wts = g * wh

wh[range(10)]
wh.sum()
new_wts.sum()

ustargs = np.dot(xmat.T, wh)
ustargsrw = np.dot(xmat.T, new_wts)
usdiffs = ustargsrw - ustargs
uspdiffs = usdiffs / ustargs * 100
np.round(uspdiffs, 2)


gw1.method
gw1.elapsed_seconds
gw1.sspd
gw1.geotargets_opt
np.round(gw1.geotargets_opt / targets * 100 - 100, 2)
gw1.whs_opt
dir(gw1)
gw1.method_result


gw3 = stub_prob.geoweight(method='poisson')
gw3.method
gw3.elapsed_seconds
gw3.geotargets_opt
gw3.whs_opt
gw3.sspd
dir(gw3.method_result)

gw2 = stub_prob.geoweight(method='qmatrix-ec')
uo = {'max_iter': 20}
so = {'objective': 'QUADRATIC'}
gw2 = stub_prob.geoweight(method='qmatrix-ec', user_options=uo)
gw2 = stub_prob.geoweight(method='qmatrix-ec', solver_options=so)
gw2.method
gw2.geotargets_opt
gw2.sspd




wh.shape
p.wh.shape

xmat.shape
p.xmat.shape

type(xmat)
type(p.xmat)

targets.shape
p.geotargets.shape
type(targets)
type(p.geotargets)


p = mtp.Problem(h=1000, s=10, k=5)
np.random.seed(1)
noise = np.random.normal(0, .0125, p.k)
noise
ntargets = p.targets * (1 + noise)
# ntargets = p.targets

prob = mw.Microweight(wh=p.wh, xmat=p.xmat, targets=ntargets, geotargets=p.geotargets)
gw1 = prob.geoweight(method='qmatrix')

uo = {'max_iter': 20}
gw1 = prob.geoweight(method='qmatrix', user_options=uo)

gw3 = prob.geoweight(method='poisson')
gw3.method
gw3.elapsed_seconds
gw3.geotargets_opt
gw3.whs_opt
dir(gw3.method_result)


# use one of the following ht2_sub_adj
# ht2stub = ht2_sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub = ht2_sub_adj.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
ht2stub
# show average target value per return times 100
round(ht2stub[targvars].div(ht2stub.nret_all, axis=0) * 100, 1)

# use one of the following
# htot = ht2sums.query('HT2_STUB ==@stub')[targvars]
htot = ht2sumsadj.query('HT2_STUB ==@stub')[targvars]

ptot = pufsums.query('HT2_STUB ==@stub')[targvars]
ptot / htot

# create an adjusted ht2stub that only has target states
ht2stub_adj = ht2stub.copy()
mask = np.logical_not(ht2stub_adj['STATE'].isin(targstates))
column_name = 'STATE'
ht2stub_adj.loc[mask, column_name] = 'XX'
ht2stub_adj[['STATE', 'HT2_STUB']].groupby(['STATE']).agg(['count'])
ht2stub_adj = ht2stub_adj.groupby(['STATE', 'HT2_STUB']).sum()
ht2stub_adj.info()
# average target value per return
round(ht2stub_adj.div(ht2stub_adj.nret_all, axis=0), 1)
ht2stub_adj.sum()
ht2stub_adj
# pufsums.query('HT2_STUB == @stub')[targvars]
# np.array(ht2stub_adj.sum())
# ratios = pufsums.query('HT2_STUB == @stub')[targvars] / np.array(ht2stub_adj.sum())
# ratios = np.array(ratios)

# create possible starting values -- each record given each state's shares
ht2shares = ht2stub_adj.loc[:, ['nret_all']].copy()
ht2shares['share'] = ht2shares['nret_all'] / ht2shares['nret_all'].sum()
ht2shares = ht2shares.reset_index('STATE')

start_values = pufstub.loc[:, ['HT2_STUB', 'pid', 'wtnew']].copy().set_index('HT2_STUB')
# cartesian product
start_values = pd.merge(start_values, ht2shares, on='HT2_STUB')
start_values['iwhs'] = start_values['wtnew'] * start_values['share']
start_values  # good, everything is in the right order

iwhs = start_values.iwhs.to_numpy()  # initial weights, households and states

wh = pufstub.wtnew.to_numpy()
xmat = np.asarray(pufstub[targvars], dtype=float)
xmat.shape
# use one of the following
# targets1 = ht2stub.drop(columns=['STATE', 'HT2_STUB'])
# targets = ht2stub_adj # .drop(columns=['STATE', 'HT2_STUB'])
targets = np.asarray(ht2stub_adj, dtype=float)
targets
targets.shape
# targets_scaled = targets * ratios
# targets.shape
# targets_scaled.shape

# targets_scaled / targets

# scale targets by ratio of pufsums to HT2



# %% geoweight
gw1 = prob.geoweight(method='qmatrix')

uo = {'max_iter': 20}
gw1 = prob.geoweight(method='qmatrix', user_options=uo)

gw1.method_result.iter_opt

# dir(gw1)
gw1.method
gw1.elapsed_seconds
gw1.geotargets_opt
gw1.whs_opt
dir(gw1.method_result)

gw2 = prob.geoweight(method='qmatrix-ec')
uo = {'max_iter': 20}
so = {'objective': 'QUADRATIC'}
gw2 = prob.geoweight(method='qmatrix-ec', user_options=uo)
gw2 = prob.geoweight(method='qmatrix-ec', solver_options=so)
gw2.method
gw2.geotargets_opt
gw2.sspd

gw3 = prob.geoweight(method='poisson')
gw3.method
gw3.elapsed_seconds
gw3.geotargets_opt
gw3.whs_opt
dir(gw3.method_result)

# sum of squared percentage differences
gw1.sspd
gw2.sspd

