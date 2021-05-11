
# TODO:
# DONE: compare geotargets to national values
# DONE: compare geotarget shares to naive expected shares
# develop measure of difficulty
# adjust targets or weights to account for difficulty

# Conclusions:
# poisson_lsq with jac, scalegoal 10.0 seems best
# poisson_lsq with findiff also good
# poisson_lsq with jvp-linop can't use x_scale='jac' so it is not good
# poisson_newton too dependent on starting point

# geoipopt is good but not terribly fast and requires good guesses re constraint ranges

# qmatrix??


# sudo sync && sudo sysctl -w vm.drop_caches=3

# %% imports
from importlib import reload

import sys
# either of the following is close but it can't find paramtols in calculator.py:
# sys.path.append('C:/programs_python/Tax-Calculator/build/lib/')  # needed
# sys.path.insert(0, 'C:/programs_python/Tax-Calculator/build/lib/') # this is close
from pathlib import Path
import os

import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date
from timeit import default_timer as timer
import pickle

import functions_advance_puf as adv
import functions_reweight_puf as rwp
import functions_geoweight_puf as gwp
import functions_ht2_analysis as fht

import puf_constants as pc
import puf_utilities as pu

# microweight - apparently we have to tell python where to find this
# sys.path.append('c:/programs_python/weighting/')  # needed
weighting_dir = Path.home() / 'Documents/python_projects/weighting'
# weighting_dir.exists()
sys.path.append(str(weighting_dir))  # needed
import src.make_test_problems as mtp
import src.microweight as mw


# %% reimports
# reload(pc)
# reload(rwp)
# reload(gwp)
reload(mw)


# %%  locations
# machine = 'windows'
machine = 'linux'

if machine == 'windows':
    DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
    DATADIR = r'C:\programs_python\puf_analysis\data/'
    # the following locations store files not saved in git
    IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
elif machine == 'linux':
    # /home/donboyd/Dropbox/PUF files/files_based_on_puf2011
    DIR_FOR_OFFICIAL_PUF = r'~/Dropbox/PUF files/files_based_on_puf2011/2020-08-20/'
    DATADIR = '/media/don/ignore/data/'
    IGNOREDIR = '/media/don/ignore/' # /media/don

PUFDIR = IGNOREDIR + 'puf_versions/'
TCOUTDIR = PUFDIR + 'taxcalc_output/'
WEIGHTDIR = PUFDIR + 'weights/'

TABDIR = IGNOREDIR + 'result_tables/'

TEMPDIR = IGNOREDIR + 'intermediate_results/'


# %% paths to specific already existing files
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'

# growfactors
GF_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'growfactors.csv'
# GF_CUSTOM = DATADIR + 'growfactors_custom.csv'  # selected growfactors reflect IRS growth between 2011 and 2017
GF_CUSTOM = DATADIR + 'growfactors_custom_busincloss.csv'  # now also includes business income and loss items
GF_ONES = DATADIR + 'growfactors_ones.csv'

WEIGHTS_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'puf_weights.csv'

POSSIBLE_TARGETS = DATADIR + 'targets2017_possible.csv'

HT2_SHARES = DATADIR + 'ht2_shares.csv'

# %% functions
def wtdsums(df, vars, weight):

    return wtdsums



# %% constants
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
compstates = ['NY', 'AR', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX']


# %% retrieve puf and targets info
pkl_name = IGNOREDIR + 'pickle.pkl'
open_file = open(pkl_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

targvars, ht2wide, pufsub, dropsdf_wide = pkl


# %% prep
wfname_national = WEIGHTDIR + 'weights2017_georwt1.csv'
wfname_national
final_national_weights = pd.read_csv(wfname_national)
# final_national_weights.head(20)

# %% counts by stub
# pufsub
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])


# %% define a stub

# get the puf data for a stub and convert to float
stub = 5
qx = '(ht2_stub == @stub)'

pufstub = pufsub.query(qx)[['pid', 'ht2_stub'] + targvars]
# pufstub.replace({False: 0.0, True: 1.0}, inplace=True)
pufstub[targvars] = pufstub[targvars].astype(float)

# get targets and national weights
targetsdf = ht2wide.query(qx)[['stgroup'] + targvars]
whdf = pd.merge(pufstub[['pid']], final_national_weights[['pid', 'weight']], how='left', on='pid')

wh = whdf.weight.to_numpy()
np.quantile(wh, qtiles)

xmat = pufstub[targvars].astype(float).to_numpy()
xmat
xmat[:, 0:7]
xmat.sum(axis=0)

geotargets = targetsdf[targvars].to_numpy()
geotargets = np.where(geotargets==0, 1e3, geotargets)
# replace any zeros with 1e3


# %% check the targets data
# how do target sums for nation compare to nationally weighted puf??
geotargets.shape
targetsums = geotargets.sum(axis=0)
targetsums

pufsums = xmat.T.dot(wh)
targetsums - pufsums
np.round(targetsums / pufsums, 3)

# vnum = 4
# targetsums[[vnum]] - pufsums[[vnum]]

# %% force geotargets to add to the national sums (although differences are not large)
factors = pufsums / targetsums
np.round(factors, 4)
imax = np.argmax(np.abs(factors - 1))
targvars[imax]  # 12 c01000 capital gains  15 c18300
factors[imax]

geotargets_adj = geotargets * factors
geotargets_adj / geotargets

geotargets_adj
targetsums_adj = geotargets_adj.sum(axis=0)
targetsums_adj

np.round(targetsums_adj - pufsums, 5)  # good
targetsums_adj / pufsums


# %% which targets seem hard to hit??
np.sort(geotargets_adj.flatten())  # no zero values
# construct initial weights
init_shares = (targetsdf.nret_all / targetsdf.nret_all.sum()).to_numpy()
targetsdf.stgroup
init_shares
Q_init = np.tile(init_shares, (wh.size, 1))
Q_init.shape

whs_init = np.multiply(Q_init.T, wh).T
whs_init.shape

geotargets_init = np.dot(whs_init.T, xmat)
target_ratios = geotargets_init / geotargets_adj
np.quantile(target_ratios, qtiles)
target_ratios.shape
target_ratios.flatten()
imax = np.argmax(target_ratios)
np.argmax(target_ratios, axis=0)
np.argmax(target_ratios, axis=1)
target_ratios.flatten()[imax]

geotargets_init.flatten()[13]
geotargets_adj.flatten()[13]
geotargets.flatten()[13]
# ans.geotargets_opt.flatten()[13]
targvars[13]

init_diffs = np.abs(target_ratios.flatten() - 1.0)



# %% define a problem
# p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
# prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=p.geotargets)

prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=geotargets_adj)


# %% test one stub directly
# prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=ngtargets)

poisson_opts = {
    'scaling': True,
    'scale_goal': 10.,
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # jac or jvp for newton; also vjp, findiff if lsq
    'quiet': True}
poisson_opts

ib = reslsq.method_result.beta_opt.flatten()
poisson_opts.update({'init_beta': ib})
np.size(ib)


poisson_opts.update({'stepmethod': 'jac'})
poisson_opts.update({'stepmethod': 'jvp'})
poisson_opts.update({'max_iter': 30})
poisson_opts.update({'scaling': True})
poisson_opts.update({'scale_goal': 1e1})
poisson_opts.update({'init_beta': 0.5})

# idea: start with lsq to get initial beta and then go from there

poisson_lsq.update({'stepmethod': 'jac'})
poisson_lsq.update({'stepmethod': 'jvp'})
poisson_lsq.update({'stepmethod': 'jvp-linop'})
poisson_lsq.update({'stepmethod': 'findiff'})
poisson_lsq.update({'scale_goal': 1e1})
poisson_lsq.update({'max_nfev': 200}) #
#  9.3020e+02 cost at 40
poisson_lsq.update({'init_beta': 0.0})
poisson_lsq.update({'ftol': 3e-2})

poisson_lsq.update({'x_scale': 'jac'})  # can't use jac for jvp-linop
poisson_lsq.update({'x_scale': 1e0})
poisson_lsq.update({'x_scale': geotargets.flatten()   / 1e6})
poisson_lsq.update({'x_scale': 1e11 / geotargets.flatten()})
poisson_lsq

reslsq = prob.geoweight(method='poisson-lsq', options=poisson_lsq)
reslsq.elapsed_seconds
reslsq.sspd
np.quantile(np.abs(reslsq.pdiff), qtiles)
np.quantile(reslsq.pdiff, qtiles)
np.quantile(reslsq.whs_opt, qtiles)

dir(reslsq.method_result)
reslsq.method_result.beta_opt.shape
reslsq.method_result.beta_opt.size
np.quantile(reslsq.method_result.beta_opt, qtiles)
np.quantile(reslsq.pdiff, qtiles)

# tensor flow jax
opts = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'objscale': 1.0,
    'method': 'BFGS',  # BFGS or LBFGS
    'max_iterations': 50,
    'max_line_search_iterations': 50,
    'num_correction_pairs': 10,  # LBFGS only
    'parallel_iterations': 1,
    'tolerance': 1e-8,
    'quiet': True}
opts.update({'method': 'BFGS'})
opts.update({'method': 'LBFGS'})
opts.update({'max_iterations': 1500})
opts.update({'max_line_search_iterations': 100})
opts.update({'parallel_iterations': 1})
opts.update({'num_correction_pairs': 100})  # LBFGS only
opts.update({'objscale': 1e-3})
opts
gwp4 = prob.geoweight(method='poisson-mintfjax', options=opts)
gwp4.elapsed_seconds
gwp4.sspd  # does not converge
dir(gwp4.method_result.result)
gwp4.method_result.result.converged
gwp4.method_result.result.num_iterations
gwp4.method_result.result.num_objective_evaluations
gwp4.method_result.result.objective_value


# now try newton method
opts = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    # 'max_iter': 20,
    'stepmethod': 'jac',  # jac or jvp for newton; also vjp, findiff if lsq
    'quiet': True}
opts.update({'stepmethod': 'jac'})
opts.update({'stepmethod': 'jvp'})
opts.update({'max_iter': 50})
opts.update({'step_mult': 0.75})
opts.update({'step_mult': 0.5})
opts.update({'step_mult': 0.25})
opts.update({'init_beta': 0.0})
opts.update({'maxp_tol': 0.01}) # max pct diff tolerance .01 is 1/100 percent
opts
gwpn = prob.geoweight(method='poisson-newton', options=opts)
gwpn.elapsed_seconds
gwpn.sspd
np.round(np.quantile(gwpn.pdiff, qtiles), 3)

# trust-exact  dogleg


# %% ipopt geo
geoipopt_opts = {
        'xlb': .1, 'xub': 10., # default 0.1, 10.0
         'crange': 0.0,  # default 0.0
         # 'print_level': 0,
        'file_print_level': 5,
        # 'scaling': True,
        # 'scale_goal': 1e3,
         # 'ccgoal': 10000,
         'addup': True,  # default is false
         'max_iter': 100,
         'linear_solver': 'ma86',  # ma27, ma77, ma57, ma86 work, not ma97
         'quiet': False}

geoipopt_opts.update({'output_file': '/home/donboyd/Documents/test4.out'})
geoipopt_opts.update({'xlb': 0.0})
geoipopt_opts.update({'xub': 1e6})
geoipopt_opts.update({'addup': False})
geoipopt_opts.update({'addup': True})
geoipopt_opts.update({'scaling': True})
geoipopt_opts.update({'scale_goal': 1e1})
geoipopt_opts.update({'crange': .01})
# geoipopt_opts.update({'crange': crange_calc * 1.})
geoipopt_opts

np.round(init_diffs, 3)*100.0
init_diffs
cr2 = init_diffs.copy()
cr2[crange_calc > 0.5] = 0.2 # np.inf
cr2[(crange_calc > .1) & (crange_calc <= 0.5)] = .035
cr2[crange_calc <= .1] = .005
np.round(cr2, 3)
geoipopt_opts.update({'crange': cr2})

geoipopt_opts.update({'max_iter': 100})
geoipopt_opts.update({'addup_range': .005})
geoipopt_opts.update({'xlb': .001})
geoipopt_opts.update({'xub': 1000.0})
geoipopt_opts

ipres = prob.geoweight(method='geoipopt', options=geoipopt_opts)
ipres.elapsed_seconds
ipres.sspd  # 9709353493.535954
ipres.geotargets_opt

np.corrcoef(reslsq.whs_opt.flatten(), ipres.whs_opt.flatten())

# array([3.09695042e-02, 6.83918768e-02, 2.58291444e-01, 4.28184185e-01,
#        1.75463618e+00, 6.92272991e+00, 2.07672319e+01, 4.18776204e+01,
#        6.51681760e+01, 1.71746580e+02, 9.85351017e+04])

np.quantile(np.abs(ipres.pdiff), qtiles)
np.quantile(np.abs(ipres.whs_opt), qtiles)
np.quantile(ipres.method_result.g, qtiles)


ipres.geotargets_opt - geotargets_adj

np.round(ipres.geotargets_opt / geotargets_adj, 2)



# %% stub run using old method


# %% info:  all stubs run
grouped = pufsub.groupby('ht2_stub')
# targvars, ht2wide, dropsdf_wide, independent=False

# choose one of the following combinations of geomethod and options
# geomethod = 'qmatrix'  # does not work well
# options = {'qmax_iter': 50,
#            'qshares': qshares }  # qshares or None

# ipopt took 31 mins
geomethod = 'qmatrix-ipopt'
options = {'qmax_iter': 50,
           'quiet': True,
           'qshares': qshares,  # qshares or None
           'xlb': 0.1,
           'xub': 100,
           'crange': .0001,
           'linear_solver': 'ma57'
           }

a = timer()
final_geo_weights = grouped.apply(gwp.get_geo_weights,
                                weightdf=final_national_weights,
                                targvars=targvars,
                                ht2wide=ht2wide,
                                dropsdf_wide=dropsdf_wide,
                                independent=False,
                                geomethod=geomethod,
                                options=options,
                                intermediate_path=TEMPDIR)
b = timer()
b - a
(b - a) / 60

final_geo_weights.sum()

# %% why doesn't stub 10 work??
temp = pufsub.query('ht2_stub == 10')  # djb .query('abspdiff > 10')
tempdrops = dropsdf_wide.query('ht2_stub == 10')
tempdrops.describe()
grouped = temp.groupby('ht2_stub')
tempout = grouped.apply(gwp.get_geo_weights,
                                weightdf=final_national_weights,
                                targvars=targvars,
                                ht2wide=ht2wide,
                                dropsdf_wide=tempdrops,
                                independent=False,
                                geomethod=geomethod,
                                options=options,
                                intermediate_path=TEMPDIR)

