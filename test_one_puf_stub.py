
# TODO:
# compare geotargets to national values
# compare geotarget shares to naive expected shares
# develop measure of difficulty
# adjust targets or weights to account for difficulty


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

# %% define a stub

pufsub
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])

# get the puf data for a stub and convert to float
stub = 2
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

vnum = 4
targetsums[[vnum]] - pufsums[[vnum]]

# %% force geotargets to add to the national sums (although differences are not large)
factors = pufsums / targetsums
factors
geotargets_adj = geotargets * factors

geotargets_adj
targetsums_adj = geotargets_adj.sum(axis=0)
targetsums_adj

targetsums_adj - pufsums  # good


# %% which targets seem hard to hit??
np.sort(geotargets_adj.flatten())  # no zero values




# %% define a problem
# p = mtp.Problem(h=1000, s=3, k=3, xsd=.1, ssd=.5, pctzero=.4)
# prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=p.geotargets)

prob = mw.Microweight(wh=wh, xmat=xmat, geotargets=geotargets_adj)


# %% test one stub directly
# prob = mw.Microweight(wh=p.wh, xmat=p.xmat, geotargets=ngtargets)

poisson_opts = {
    'scaling': True,
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # jac or jvp for newton; also vjp, findiff if lsq
    'quiet': True}
poisson_opts

poisson_opts.update({'stepmethod': 'jac'})
poisson_opts.update({'max_iter': 20})

ans = prob.geoweight(method='poisson-newton', options=poisson_opts)
ans.elapsed_seconds
ans.sspd
np.quantile(np.abs(ans.pdiff), qtiles)
ans.geotargets_opt
ans.whs_opt

poisson_lsq = {
    'scaling': True,
    'scale_goal': 1e3,
    'init_beta': 0.5,
    'stepmethod': 'findiff',  # jac or jvp for newton; also vjp, findiff if lsq
    'quiet': True}
poisson_lsq

poisson_lsq.update({'stepmethod': 'jac'})

reslsq = prob.geoweight(method='poisson-lsq', options=poisson_lsq)
reslsq.elapsed_seconds
reslsq.sspd
np.quantile(np.abs(reslsq.pdiff), qtiles)


# %% ipoopt geo
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

geoipopt_opts.update({'output_file': '/home/donboyd/Documents/test6.out'})
geoipopt_opts.update({'addup': False})
geoipopt_opts.update({'addup': True})
geoipopt_opts.update({'scaling': True})
geoipopt_opts.update({'scale_goal': 1e6})
geoipopt_opts.update({'crange': .02})
geoipopt_opts.update({'max_iter': 100})
geoipopt_opts.update({'addup_range': .005})
geoipopt_opts.update({'xlb': .001})
geoipopt_opts.update({'xub': 100.0})
geoipopt_opts

ipres = prob.geoweight(method='geoipopt', options=geoipopt_opts)
ipres.elapsed_seconds
ipres.sspd
ipres.geotargets_opt

np.quantile(np.abs(ipres.pdiff), qtiles)
np.quantile(np.abs(ipres.whs_opt), qtiles)
np.quantile(ipres.method_result.g, qtiles)


ipres.geotargets_opt - targmat

ipres.geotargets_opt / targmat



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

