# https://stackoverflow.com/questions/21868369/pycharm-hanging-for-a-long-time-in-ipython-console-with-big-data
# to (maybe) fix pycharm hanging:
# Files -> Settings -> Build, Execution, Deployment -> Python Debugger
# switch on the "Gevent Compatible" flag

# installing latest Tax-Calculator or different versions (but not modifying source)
# see this video https://www.youtube.com/watch?v=c6IgzwFRf5I
# then
# conda install requests
# conda install -c conda-forge paramtools
# pip install git+https://github.com/PSLmodels/Tax-Calculator
# or
# pip install -e git+https://github.com/PSLmodels/Tax-Calculator#egg=taxcalc

# pip freeze (useful to see packages and versions installed; conda also has command)


# instructions from my conversation with Matt 11/28/2020 about installing
# Tax-Calculator from sournce on my own machine
#
# after revision of source or checkout of new version
# from command prompt window, in Tax-Calculator directory
# if there is an old one in place:
# pip uninstall taxcalc
# then:
# python setup.py install

# in terminal:
#    export OMP_NUM_THREADS=10
#    export NUMBA_NUM_THREADS=10
#


# %% about this program

# this program does the following:
from timeit import default_timer as timer
import src.microweight as mw
# create unweighted national puf for 2017 using custom growfactors for 2011 to 2017
# create weights for this 2017 national puf to come close to IRS national targets
# create tentative state weights for this regrown-reweighted national puf, without constraining them to national record weights
# use sum of tentative state weights as initial weights for 2nd round of national reweighting
# create new national reweights starting from the sums of state weights
# create state weights where sums are constrained to these new national weights
# advance this 2017 file to 2018 using default puf growfactors for 2017 to 2018 and estimate of weight growth

# the resulting 2018-income puf is used by puf_tax_analysis.py for NY project policy simulations

# it does NOT do target preparation needed for this program; for that, see the following:
# get IRS national reported values for targeting -- see puf_download_national_target_files.py
# create new (combination) IRS variables -- also see puf_download_national_target_files.py
# map puf variables to IRS targets -- see puf_ONETIME_create_puf_irs_mappings_and_targets.py
# get IRS Historical Table 2 state values to be used in state targeting -- see puf_download_state_HT2target_files.py
# create new (combination) HT2 variables and get each state's share of the total -- see puf_ht2_shares.py


# %% imports
from importlib import reload

import sys
# either of the following is close but it can't find paramtols in calculator.py:
# sys.path.append('C:/programs_python/Tax-Calculator/build/lib/')  # needed
# sys.path.insert(0, 'C:/programs_python/Tax-Calculator/build/lib/') # this is close
from pathlib import Path
from collections import namedtuple
import os
import pickle
from collections import OrderedDict

import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import functions_advance_puf as adv
import functions_puf_analysis as fpa
import functions_reports as rpt
import functions_reweight_puf as rwp
import functions_geoweight_puf as gwp
import functions_ht2_analysis as fht
import functions_state_weights as fsw

import puf_constants as pc
import puf_utilities as pu


# microweight - apparently we have to tell python where to find this
# sys.path.append('c:/programs_python/weighting/')  # needed
WEIGHTING_DIR = Path.home() / 'Documents/python_projects/weighting'
# WEIGHTING_DIR.exists()
sys.path.append(str(WEIGHTING_DIR))  # needed
import src.microweight as mw

from timeit import default_timer as timer


# %% reimports
reload(adv)
reload(fsw)
reload(fpa)
reload(gwp)
reload(mw)
reload(pc)
reload(rpt)
reload(rwp)
# reload(gwp)


# %% physical locations
# files that were created in Windows version of this
WINDATADIR = '/media/don/ignore/data/'
DIR_FOR_OFFICIAL_PUFCSV = r'/media/don/data/puf_files/puf_csv_related_files/PSL/2020-08-20/'
DIR_FOR_BOYD_PUFCSV = r'/media/don/data/puf_files/puf_csv_related_files/Boyd/2021-07-02/'


# working storage
SCRATCHDIR = '/media/don/scratch/'
OUTDIR = '/media/don/pufanalysis_output/'

# %%  relative locations to use

# input locations
# PUFDIR = DIR_FOR_OFFICIAL_PUFCSV
# WEIGHTDIR = DIR_FOR_OFFICIAL_PUFCSV

PUFDIR = DIR_FOR_BOYD_PUFCSV
WEIGHTDIR = DIR_FOR_BOYD_PUFCSV

# output locations
OUTDATADIR = OUTDIR + 'data/'
OUTTABDIR = OUTDIR + 'result_tables/'
OUTWEIGHTDIR = OUTDIR + 'weights/'
OUTSTUBDIR = OUTDIR + 'stub_output/'


# %% paths to specific already existing files
PUF_USE = PUFDIR + 'puf.csv'
GF_USE = PUFDIR + 'growfactors.csv'
WEIGHTS_USE = WEIGHTDIR + 'puf_weights.csv'
RATIOS_USE = PUFDIR + 'puf_ratios.csv'


# paths to target files previously created in other programs

# created in:  puf_ONETIME_create_puf_irs_mappings_and_targets.py
POSSIBLE_TARGETS = OUTDATADIR + 'targets2017_possible.csv'

# created in: puf_ht2_shares.py
# windows version is in /media/don/ignore/data
HT2_SHARES = WINDATADIR + 'ht2_shares.csv'  # rebuild this


# %% names of files to create
PUF_DEFAULT = OUTDATADIR + 'puf2017_default.parquet'
# PUF_REGROWN = OUTDATADIR + 'puf2017_regrown.parquet'  # not doing regrowing


# %% constants
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% BEGIN

# %% 1. Download & parse IRS summary data for national & state targets

# National:
#   puf_download_national_target_files.py
#   puf_ONETIME_create_puf_irs_mappings_and_targets.py

# State:
#   puf_download_state_HT2target_files.py
#   puf_ht2_shares.py


# %% 2. Preliminary preparation of national data file
# %% ..2.1 Advance puf.csv to a future year, save as parquet file
# add pid and filer, and save as puf+str(year).parquet
# advance official puf.csv with official weights, grow factors, and ratios
fsw.advance_and_save_puf(
    year=2017,
    pufpath=PUF_USE,
    growpath=GF_USE,
    wtpath=WEIGHTS_USE,
    ratiopath=RATIOS_USE,
    outdir=OUTDATADIR)

# Alternative: advance by "regrowing" with custom growfactors and without puf_ratios
# Not currently implemented

# %% ..2.2 Get potential national targets for 2017, previously created
# from common_stub	incrange	pufvar	irsvar	irs	table_description	column_description	src	excel_column
# should have variables:
#   common_stub	incrange, pufvar, irsvar, irs, table_description, column_description, src	excel_column
ptargets = fsw.get_potential_national_targets(
    targets_fname=POSSIBLE_TARGETS)


# %% ..2.3 Create puf subset and examine quality of puf with initial weights
# -- pufsub is subset of filers, with just those variables needed for potential targets
# from puf{year}.parquet file; only includes filer records
# adds pid, filer, stubs, and target variables
pufsub = fsw.prep_puf(OUTDATADIR + 'puf2017.parquet', ptargets)

# %% ..2.4 Examine how close initial data are to IRS targets
# % get differences from targets at initial weights and produce report
weights_initial = fsw.get_pufweights(
    wtpath=WEIGHTS_USE, year=2017)  # adds pid and shortname

pdiff_init = rwp.get_pctdiffs(pufsub, weights_initial, ptargets)
np.round(np.nanquantile(pdiff_init.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_init,
    outfile=OUTTABDIR + 'baseline_national.txt',
    title='2017 puf values using official weights, growfactors, and puf_ratios versus IRS targets.')


# %% ..2.5 Reweight national puf to come closer to targets
drops_national = fpa.get_drops_national(pdiff_init)
weights_reweight = rwp.puf_reweight(
    pufsub, weights_initial, ptargets, method='ipopt', drops=drops_national)


# %% ..2.6 Examine quality of reweighted puf
pdiff_reweighted = rwp.get_pctdiffs(pufsub, weights_reweight, ptargets)
np.round(np.nanquantile(pdiff_reweighted.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_reweighted,
    outfile=OUTTABDIR + 'reweighted_national.txt',
    title='Reweighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_init)


# %% 3. Prepare state target data

# %% ..3.1 Define states to target
# choose or modify ONE of the following
# compstates = ['NY', 'AR', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX']
# see pc.STATES, STATES_DCPROA, STATES_DCPROAUS
# compstates = pc.STATES[0:40]
compstates = pc.STATES

# %% ..3.2 Calculate state targets for states of interest
# Collapse non-targeted states into other category, and apply shares to puf sums

# # Prepare state targets and drop combinations for the states of interest
# get df with ht2 shares
ht2targets = fpa.get_potential_state_targets(
    pufsub,
    weightdf=weights_reweight,
    ht2sharespath=HT2_SHARES,  # currently shares are from the old Windows run, need to update
    compstates=compstates)

# %% ..3.3 Examine quality
# report on:
#  (a) any HT2 state shares that don't add to 100%
#  (b) HT2 targets vs. puf targets
rpt.ht2_vs_puf_report(
    ht2targets,
    outfile=OUTTABDIR + 'ht2vspuf_targets_national.txt',
    title='Comparison of PUF targets, PUF values, and Historical Table 2 values',
    outpath=OUTDATADIR + 'ht2_vs_puf.csv')  # write csv file with ht2 vs puf national totals

# report on each state's share for every pufvar-ht2_stub combination
#   compared to its share of returns (e.g., in most/all stubs, NY's share
#   of SALT likely will be greater than its share of returns)
rpt.ht2target_report(
    ht2targets,
    outfile=OUTTABDIR + 'ht2target_analysis.txt',
    title='Comparison of Historical Table 2 shares by group to shares for # of returns',
    outpath=OUTDATADIR + 'state_shares.csv')

# %% ..3.4 Compute potential drops and a reformatted targets file
drops_states = fpa.get_drops_states(ht2targets)
ht2wide = fpa.get_ht2wide_states(ht2targets)


# %% 4.Prepare national PUF data for state targeting

# %% ..4.1 Define which subset of the possible state targets we will target
# targvars are the variables we will target -- must be in ht2_possible
targvars = ['nret_all', 'mars1', 'mars2',  # num returns total and by filing status
            'c00100',   # AGI
            'e00200', 'e00200_nnz',  # wages
            'e00300', 'e00300_nnz',  # taxable interest income
            'e00600', 'e00600_nnz',  # ordinary dividends
            'e00900',  # business and professional income
            'e26270',  # partnership/S Corp income
            # added c01000_nnz 7/7/2021
            # capital gains (tc doc says see .py file, though)
            'c01000', 'c01000_nnz',
            'c02500', 'c02500_nnz',  # taxable Social Security added 7/7/2021
            # deductions
            'c17000', 'c17000_nnz',  # medical expenses deducted
            'c19700', 'c19700_nnz',  # contribution deductions added 7/7/2021
            'c18300', 'c18300_nnz']  # SALT amount deducted

# for testing purposes, here are some useful subsets of targvars
# targvars2 = ['nret_all']
# targvars2 = ['nret_all', 'c00100']
# targvars2 = ['nret_all', 'c00100', 'e00200']
# targvars2 = ['nret_all', 'mars1', 'c00100']
# targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']
# targvars2 = ['nret_all', 'c00100', 'e00200', 'c18300']

# verify that targvars are in the data
len(targvars)
['good' for var in targvars if var in ht2wide.columns]

# scratch area: identifying all-zero variables
sub = ht2wide.loc[ht2wide.ht2_stub == 1, :]
good = sub.loc[:, (sub.sum(axis=0) != 0)]
# dropcols = sub.columns.difference(good.columns).tolist()
dropcols = [var for var in targvars if not var in good.columns]
keepcols = [var for var in targvars if var in good.columns]
keepcols


# %% ..4.2. Construct national weights as sums of unrestricted state weights

# how many records in each HT2 stub? ranges from 5,339 in stub 1 to 41,102 in stub 4
# pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])

# now we are going to iterate through the HT2 agi ranges (grouped.apply)
#   and within each range iterate through the states  (gwp.get_geo_weights)
# because we are NOT imposing an adding-up restriction:
#  independent=True
# this will be fairly fast
# nat_geo_weights will have 1 record per return, with id info plus:
# weight: the initial national weight for the record
# geoweight_sum: the record's sum of the weights over the solved-for
# a column for each solved-for state, with the record's weight for the state
# while geoweight_sum will not equal weight (our initial national weight), for
# most records it will be quite close


# NOTE: this next statement takes about 11 minutes
# weights_geosums only has the sum-of-states weight for each record
# the 50 state weights for each record are in the allweights... csv file
weights_geosums = gwp.get_geoweight_sums(
    pufsub,
    weightdf=weights_reweight,
    targvars=targvars,
    ht2wide=ht2wide,
    dropsdf_wide=drops_states,
    outpath=OUTWEIGHTDIR + 'allweights2017_geo_unrestricted.csv',
    stubs=None)  # None, or list or tuple of stubs

# To get weights_geosums from the file, run the following 2 lines
# weights_geosums = pd.read_csv(OUTWEIGHTDIR + 'allweights2017_geo_unrestricted.csv')
# weights_geosums = weights_geosums.loc[:,['pid', 'geoweight_sum']].rename(columns={'geoweight_sum': 'weight'})

# %% ..4.3 Examine quality of tentative national weights that are sums of unrestricted state weights
pdiff_geosums = rwp.get_pctdiffs(pufsub, weights_geosums, ptargets)
np.round(np.nanquantile(pdiff_geosums.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_geosums,
    outfile=OUTTABDIR + 'geosums_national.txt',
    title='Unrestricted geosum weighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_reweighted)


# %% ..4.3.1 Prepare data for report on unrestricted state weights (takes a while)
allweights2017_geo_unrestricted = pd.read_csv(
    OUTWEIGHTDIR + 'allweights2017_geo_unrestricted.csv')

a = timer()
vars = pufsub.columns.to_list()
calcvars = [x for x in vars if x not in [
    'pid', 'filer', 'common_stub', 'ht2_stub']]
rpt.calc_save_statesums(
    pufsub,
    state_weights=allweights2017_geo_unrestricted,
    pufvars=calcvars,
    outfile=OUTDATADIR + 'state_sums_wunrestricted.csv')
b = timer()
b - a  # ~ 4 mins

# %% ..4.3.2 Report on quality of these unrestricted state weights
rpt.state_puf_vs_targets_report(
    state_targets=ht2targets,
    state_sums=OUTDATADIR + 'state_sums_wunrestricted.csv',
    title='State calculated values vs. state targets',
    reportfile=OUTTABDIR + 'state_comparison_wunrestricted.txt'
)

# %% ..4.4 Reweight these tentative national weights to come closer to IRS targets
drops_national_geo = fpa.get_drops_national(pdiff_geosums)
weights_georeweight = rwp.puf_reweight(
    pufsub, weights_geosums, ptargets, method='ipopt', drops=drops_national_geo)

# %% ..4.5 Examine quality of the reweighted sums-of-unrestricted-state-weights
pdiff_georeweighted = rwp.get_pctdiffs(pufsub, weights_georeweight, ptargets)
np.round(np.nanquantile(pdiff_georeweighted.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_georeweighted,
    outfile=OUTTABDIR + 'geosums_reweighted_national.txt',
    title='Geosum reweighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_geosums)


# %% ..4.6 Update state-stub targets to reflect new slightly-revised pufsums, report on quality
# remember that new pufsums won't be exactly like old so we should recalibrate
# state targets
ht2targets_updated = fpa.get_potential_state_targets(
    pufsub,
    weightdf=weights_georeweight,
    ht2sharespath=HT2_SHARES,  # currently this is from the old Windows run, need to update
    compstates=compstates)

# report on:
#  (a) any HT2 state shares that don't add to 100%
#  (b) HT2 targets vs. puf targets
rpt.ht2_vs_puf_report(
    ht2targets_updated,
    outfile=OUTTABDIR + 'ht2vspuf_targets_national_updated.txt',
    title='Comparison of PUF targets, PUF values with reweighted-geosums, and Historical Table 2 values',
    outpath=OUTDATADIR + 'ht2_vs_puf_updated.csv')  # write csv file with ht2 vs puf national totals

# report on each state's share for every pufvar-ht2_stub combination
#   compared to its share of returns (e.g., in most/all stubs, NY's share
#   of SALT likely will be greater than its share of returns)
rpt.ht2target_report(
    ht2targets,
    outfile=OUTTABDIR + 'ht2target_analysis_updated.txt',
    title='Comparison of Historical Table 2 shares by group to shares for # of returns',
    outpath=OUTDATADIR + 'state_shares_updated.csv')

drops_states_updated = fpa.get_drops_states(ht2targets_updated)
ht2wide_updated = fpa.get_ht2wide_states(ht2targets_updated)


# %% ..4.7 Pickle results to this point that we will need for creating state weights
# to avoid running all of the code above each time we test state weighting,
# pickle the data needed for state weighting once and retrieve when needed
save_list = [
    pufsub,
    ptargets, ht2targets, ht2targets_updated,
    ht2wide, ht2wide_updated,
    # national weights
    weights_initial, weights_reweight, weights_georeweight,
    weights_geosums,
    compstates,
    targvars,
    drops_states_updated]
save_name = SCRATCHDIR + 'pufsub_state_weighting_package.pkl'

open_file = open(save_name, "wb")
pickle.dump(save_list, open_file)
open_file.close()





# %% 5. Get state weights

# %% ..5.1 Retrieve pickled data for state weighting
save_name = SCRATCHDIR + 'pufsub_state_weighting_package.pkl'
open_file = open(save_name, "rb")
pkl = pickle.load(open_file)
open_file.close()

pufsub, ptargets, ht2targets, ht2targets_updated, \
    ht2wide, ht2wide_updated, \
    weights_initial, weights_reweight, weights_georeweight, weights_geosums, \
    compstates, targvars, drops_states_updated = pkl
del(pkl)

# %% ..5.2. Loop through stubs and save results
# %% ..5.2.1 Choose stub(s) to run

stubs = (1,)
stubs = (2,)
stubs = (3,)
stubs = (4,)
stubs = (5,)
stubs = (6,)
stubs = (7,)
stubs = (8,)
stubs = (9,)
stubs = (10,)
stubs = tuple(range(1, 11))
stubs = tuple(range(2, 11))

stubs = tuple((1, tuple(range(3, 11))))
stubs = (1, 3, 4, 5, 6, 7, 8, 9, 10)

# %% ..5.2.2 Define options for the run
opts = {}
opts['method_names'] = ('jac', 'krylov', 'jvp')
opts['method_maxiter_values'] = (20, 1000, 5)
opts['method_improvement_minimums'] = (0.05, 1e-9, 0.001)
opts['jac_lgmres_maxiter'] = 30
opts['jvp_lgmres_maxiter'] = 30

# opts['method'] = ('krylov',)
opts['max_search_iter'] = 30  # 20 default
opts['krylov_tol'] = 1e-9  # 1e-3
opts['pbounds'] = (.0001, 1.0)
opts['notes'] = True
opts['notes'] = False
opts['maxseconds'] = 10 * 60

opts['method_names'] = ('krylov',)
opts['method_maxiter_values'] = (1000,)
opts['method_improvement_minimums'] = (1e-4,)

opts['method_names'] = ('jac',)
opts['method_maxiter_values'] = (100,)

opts['method_names'] = ('jac', 'jvp')
opts['method_maxiter_values'] = (100, 10)

opts['method_names'] = ('krylov', 'jac')
opts['method_maxiter_values'] = (1000, 10)

opts['method_names'] = ('jac', 'krylov')
opts['method_maxiter_values'] = (20, 1000)

opts['method_names'] = ('jac', 'jvp',  'krylov')
opts['method_maxiter_values'] = (10, 5, 1000)

opts['method_names'] = ('jvp',)
opts['method_maxiter_values'] = (100,)

# {'method_names': ('krylov', 'jac'),
#  'method_maxiter_values': (1000, 10),
#  'method_improvement_minimums': (1e-06,),
#  'krylov_tol': 1e-09,
#  'pbounds': (0.0001, 1.0),
#  'notes': False,
#  'max_search_iter': 30,
#  'maxseconds': 600,
#  'jac_lgmres_maxiter': 30,
#  'jvp_lgmres_maxiter': 30}

# opts['method'] = 'poisson-newton'

# %% ..5.2.3 Run the stub(s)
gwp.runstubs(
    stubs,
    pufsub,
    weightdf=weights_georeweight,
    targvars=targvars,
    ht2wide=ht2wide_updated,
    dropsdf_wide=drops_states_updated,
    approach='poisson-newton',  # poisson-newton poisson-root
    options=opts,
    outdir=SCRATCHDIR,  # OUTSTUBDIR SCRATCHDIR
    write_logfile=True,  # boolean
    parallel=False)  # boolean

# %% ..5.3 Assemble file of weights from individual stubs
def f(stub):
    fname = OUTSTUBDIR + 'stub' + str(stub).zfill(2) + '_whs.csv'
    df = pd.read_csv(fname)
    return df

frames = [f(stub) for stub in range(1, 11)]

allweights2017_geo_restricted = pd.concat(frames).sort_values(by='pid')
allweights2017_geo_restricted

allweights2017_geo_restricted.to_csv(
    OUTWEIGHTDIR + 'allweights2017_geo_restricted.csv', index=False)
del(frames)


# %% ..5.4 Examine quality of state optimization results
# %% ..5.4.1. Summarize puf by state and ht2_stub and save

a = timer()
vars = pufsub.columns.to_list()
calcvars = [x for x in vars if x not in [
    'pid', 'filer', 'common_stub', 'ht2_stub']]
rpt.calc_save_statesums(
    pufsub,
    state_weights=allweights2017_geo_restricted,
    pufvars=calcvars,
    outfile=OUTDATADIR + 'state_sums_wrestricted.csv')
b = timer()
b - a  # ~ 4 mins


# %% ..5.4.2 Report on quality

# reload(rpt)
rpt.state_puf_vs_targets_report(
    state_targets=ht2targets_updated,
    state_sums=OUTDATADIR + 'state_sums_wrestricted.csv',
    title='State calculated values vs. state targets',
    reportfile=OUTTABDIR + 'state_comparison_wrestricted.txt'
)


# %% 6. Advance file to years after 2017

# TBD

# %% DEADWOOD AND MISCELLANEOUS ISSUES UNDER EXPLORATION

# niter = 0
# l2norm_best = 1e99
# beta_best = 0.0

# ibeta = np.load(OUTSTUBDIR + 'stub02_betaopt.npy').flatten()
# ibeta.shape
# ibeta = beta_best.copy()
# l2b = l2norm_best.copy()


# %% opts-set
opts = opts_andy
opts = opts_b1
opts = opts_b2
opts = opts_dfs
opts = opts_jfnk
opts = opts_kry
opts = opts_lm
opts = opts_lm2
opts = opts_newt

opts

opts.update({'scale_goal': 1e3})
opts.update({'scaling': True})


# %% get parquet file

tmp = pd.read_parquet(OUTDATADIR + 'puf2017.parquet', engine='pyarrow')



# %% callback function
def callback(x, f):
    # x is solution
    # f are residuals
    # set niter = 0 outside this function, before running optimization
    global niter, beta_best, l2norm_best
    l2norm = np.linalg.norm(f, 2)
    if l2norm < l2norm_best:
        l2norm_best = l2norm
        beta_best = x
    maxpdiff = np.max(np.abs(f))
    print(
        f'iter: {niter: 5};  l2norm: {l2norm: 9.2f};  max abs diff: {maxpdiff: 9.3f}')
    niter += 1
    return


# opts_lsq = {
#     'scaling': True,
#     'scale_goal': 1e1,
#     'init_beta': 0.5,
#     'stepmethod': 'jac',  # jac or jvp for newton; also vjp, findiff if lsq
#     'max_nfev': 200,  # 100 default
#     'quiet': True}
# opts_lsq.update({'max_nfev': 1000})
# opts = opts_lsq; method = 'poisson-lsq'

# %% ..10a. Setup options for poisson-newton

# method='poisson-newton'

# opts = {
#     'base_stepmethod': 'jac',  # jac default, or jvp, jac faster/better but less robust
#     'init_beta': 0.0,  # 0.0 default, can be scalar or vector
#     'jac_min_improvement': 0.10,  # .10 default, min proportionate improvement in l2norm to continue with jac
#     'jac_threshold': 5,  # default 5.0; try to use jac when rmse is below this
#     'jvp_reset_steps': 5,  # 5 default, num of jvp steps to do before retrying jac step
#     'lgmres_maxiter': 20, # 20 default, maxiter for solving for jvp step
#     'max_iter': 20,  # 20 default
#     'maxp_tol': .01,  # .01 default, 1/100 of 1% max % difference from target
#     'scaling': True,  # True default
#     'scale_goal': 10.0,  # 10.0 default, goal for sum of each scaled column of xmat
#     'search_iter': 20,  # 20 default steps for linesearch optimization
#     'stepmethod': 'auto',
#     'quiet': True}

# # work area for modifying options
# # best options
# opts.update({'jac_threshold': 1e9})
# opts.update({'jac_min_improvement': 0.10})
# opts.update({'jvp_reset_steps': 4})
# opts.update({'lgmres_maxiter': 30})
# opts.update({'max_iter': 40})
# opts.update({'no_improvement_proportion': 1e-3})
# opts.update({'search_iter': 20})
# opts.update({'stepmethod': 'auto'})

# OrderedDict(sorted(opts.items()))

# opts.update({'jac_min_improvement': 0.03})
# opts.update({'max_iter': 30})
# opts.update({'lgmres_maxiter': 20})
# opts.update({'search_iter': 50})


# %% ..10a. Verify that individual stubs can run
# res = gwp.get_geo_weights_stub(
#     pufsub,
#     weightdf=weights_georeweight,
#     targvars=targvars,  # use targvars or a variant targstub1 targvars2
#     ht2wide=ht2wide_updated,
#     dropsdf_wide=drops_states_updated,
#     method=method,  # poisson-lsq, poisson-newton, poisson-newton-sep
#     options=opts,
#     stub=1)
# res._fields
# res.elapsed_seconds
# res.whsdf
# res.beta_opt
# # compare results to targets for a single stub
# beta_save = res.beta_opt

# stub 1    5,340; drops 4 zero HT2 sum variables
# stub 2   19,107; cannot reach zero
# stub 3   35,021;
# stub 4   40,940;
# stub 5   25,992;
# stub 6   18,036;
# stub 7   30,369;
# stub 8   17,768;
# stub 9   12,504;
# stub 10  28,433; cannot reach zero; replaces 40 of 867 targets that are zero

# note that stubs 2 and 10 don't solve to zero, but the others do

# targs_used = targvars  # targsstub1 targvars2 targvars
# stub = 1

# df = pufsub.loc[pufsub['ht2_stub']==stub, ['pid', 'ht2_stub'] + targs_used]
# htstub = ht2wide_updated.loc[ht2wide_updated['ht2_stub']==stub, ['ht2_stub', 'stgroup'] + targs_used]

# sts = htstub.stgroup.tolist()  # be sure that target rows and whs columns are in sts order
# xmat = np.asarray(df[targs_used], dtype=float)
# targmat = np.asarray(htstub.loc[:, targs_used])
# targmat.size
# np.count_nonzero(targmat)
# whs = np.asarray(res.whsdf.loc[res.df['ht2_stub']==stub, sts], dtype=float)
# np.quantile(whs, qtiles)

# targopt = np.dot(whs.T, xmat)
# diff = targopt - targmat
# pdiff = diff / targmat * 100
# sspd = np.square(pdiff).sum()
# sspd
# np.round(np.quantile(pdiff, qtiles), 2)
# np.round(np.nanquantile(pdiff, qtiles), 2)
# %% ..10c. ALT setup df-sane
# opts = {
#     'scaling': True,
#     'scale_goal': 1e1,
#     'init_beta': 0.0,
#     'quiet': True}
# opts
opts_andy = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,
    'quiet': True,
    'solver': 'anderson',
    'jac': None,
    'callback': callback,
    'solver_opts': {
        'disp': False,
        'maxiter': 2000,
        'line_search': 'wolfe',  # armijo, wolfe, None
        'jac_options': {
            #  'alpha': 1e6,
            'M': 50,  # 5
            'w0': .75
        }
    }
}
opts_andy


# %% opts_b1
opts_b1 = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,
    'solver': 'broyden1',
    'jac': None,
    'callback': callback,
    'solver_opts': {
        # 'disp': True,  # |F(x)| is max abs diff
        'maxiter': 1000,
        'line_search': 'wolfe',  # armijo, wolfe, None
        'jac_options': {
            #  'alpha': -0.1
            'reduction_method': 'svd',  # restart, simple, svd
            # 'max_rank': 1e3  # infinity
        }
    }
}
opts_b1

# %% opts_b2
opts_b2 = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,
    'solver': 'broyden2',
    'jac': None,
    'callback': callback,
    'solver_opts': {
        # 'disp': True,  # |F(x)| is max abs diff
        'maxiter': 2000,
        'line_search': 'wolfe',  # armijo, wolfe, None
        'jac_options': {
            'reduction_method': 'svd',  # restart, simple, svd
            'max_rank': 100,  # infinity
        }
    }
}
opts_b2

# %% opts_dfs
opts_dfs = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,
    'solver': 'df-sane',
    'jac': None,
    'callback': callback,
    'solver_opts': {
        'disp': False,  # boolean
        'maxfev': 6000,  # `1000
        'sigma_eps': .00001,  # 1e-10, 5 best but doesn't make sense
        'sigma_0': -1.0,   # 1.0
        'M': 30,  # 10
        'line_search': 'cruz'  # cruz, cheng
    }
}
opts_dfs

# %% opts_jfnk
opts_jfnk = {
    'method': 'poisson-jfnk',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,
    'callback': callback,
    'maxiter': 200,
    'line_search': 'armijo',  # armijo, wolfe, None
    'verbose': True
}
opts_jfnk


# %% opts_kry
opts_kry = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,  # 0.0,
    'solver': 'krylov',
    'jac': None,
    'callback': callback,
    'solver_opts': {
        # 'disp': True,
        'maxiter': 530,
        'fatol': 1e-2,  # 6e-6
        'xatol': 1e-2,
        'line_search': 'wolfe',  # armijo, wolfe, None
        'jac_options': {
            # 'inner_M': 'kjac',
            'rdiff': 1e-8,  # not sure default
            'inner_maxiter': 100,  # 30
            'method': 'lgmres'
        }
    }
}
opts_kry

# %% opts_lm
opts_lm = {
    'method': 'poisson-root',
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.0,  # 0.0,
    'solver': 'lm',
    'jac': 'jac',  # False, jac
    'callback': None,  # lm cannot use callback function
    'solver_opts': {
        'maxiter': 100,  # 100*(N+1)
        'factor': 10,  # 100  5 mins, 10 is 3 mins, 1 is 2.5 mins
        # 'ftol': 1e-6,  # relative error desired in the sum of squares
        # 'xtol': 1e-10,  # 1.49e-8 relative error desired in the approximate solution
        # 'eps': 1e-10,
        # 'epsfcn': 1e-1, # suitable step length forward diff
        'col_deriv': False
    }
}
opts_lm

opts_lm2 = {
    'method': 'poisson-lsq',
    'scaling': True,
    'scale_goal': 10,
    'init_beta': 0.0,
    'jac': 'jac',  # False, jac
    'stepmethod': 'jac',  # vjp, jvp, full, finite-diff
    'max_nfev': 30,
    'gtol': 1e-2,
    'x_scale': 'jac',
    'callback': None,  # lm cannot use callback function
}
opts_lm2


# %% opts_newt
opts_newt = {
    'method': 'poisson-newton',
    'max_iter': 40,  # 20 default
    'maxp_tol': .01,  # .01 default, 1/100 of 1% max % difference from target
    'init_beta': 0.0,  # 0.0,  # 0.0 default, can be scalar or vector
    'scaling': True,  # True default
    'scale_goal': 10.0,  # 10.0 default, goal for sum of each scaled column of xmat
    'stepmethod': 'auto',
    'search_iter': 30,  # 20 default steps for linesearch optimization
    'base_stepmethod': 'jac',  # jac default, or jvp, jac faster/better but less robust
    # .10 default, min proportionate improvement in l2norm to continue with jac
    'jac_min_improvement': 0.05,
    'jvp_precondition': False,
    'jvp_reset_steps': 4,  # 5 default, num of jvp steps to do before retrying jac step
    'jac_threshold': 1e9,  # default 5.0; try to use jac when rmse is below this
    'lgmres_maxiter': 30,  # 20 default, maxiter for solving for jvp step
    'no_improvement_proportion': 1e-3,
    'notes': False
}
opts_newt

# best options
opts_newt.update({'base_stepmethod': 'jvp'})
opts_newt.update({'jac_threshold': 0})
opts_newt.update({'jac_min_improvement': 100.25})
opts_newt.update({'jvp_reset_steps': 4})
opts_newt.update({'jvp_precondition': False})
opts_newt.update({'lgmres_maxiter': 60})
opts_newt.update({'max_iter': 40})
opts_newt.update({'no_improvement_proportion': 1e-3})
opts_newt.update({'search_iter': 30})
opts_newt.update({'stepmethod': 'auto'})


# test options
opts_newt.update({'base_stepmethod': 'jvp'})
opts_newt.update({'jac_threshold': 0})
opts_newt.update({'jac_min_improvement': 100.0})
opts_newt.update({'jvp_precondition': False})
opts_newt.update({'jvp_reset_steps': 2})
opts_newt.update({'notes': True})
opts_newt.update({'init_beta': 0.5})

opts_newt.update({'scale_goal': 1e6})
opts_newt.update({'scaling': True})

# GOOD!:
# stub2 26.5319 sspd:  1189.6837829245314
# Elapsed minutes:  5.51
# opts.update({'solver': 'krylov', 'jac': None,
#   'solver_opts': {
#       'disp': True,
#       'maxiter': 300,
#       'fatol': 1e-4,  # 6e-6
#       'jac_options': {
#           # 'inner_M': 'kjac',
#           'rdiff': 1e-5,
#           'inner_maxiter': 80,
#           'method': 'lgmres'},  # lgmres
#       'line_search': 'wolfe'  # armijo, wolfe, None
#       }})


opts.update({'init_beta': 0.0})
opts.update({'init_beta': ibeta})
opts.update({'jac_min_improvement': 0.0})
opts.update({'max_iter': 2000})
opts_newt.update({'base_stepmethod': 'jac'})
opts_newt.update({'jac_threshold': 1e9})

# opts.update({'line_search': 'wolfe'})  # armijo
# opts.update({'line_search': 'armijo'})  # armijo