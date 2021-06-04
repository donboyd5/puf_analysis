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


# %% about this program

# this program does the following:
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

import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import functions_advance_puf as adv
import functions_reports as rpt
import functions_reweight_puf as rwp
import functions_geoweight_puf as gwp
import functions_ht2_analysis as fht
import functions_state_weights as fsw

import puf_constants as pc
import puf_utilities as pu

import test

# microweight - apparently we have to tell python where to find this
# sys.path.append('c:/programs_python/weighting/')  # needed
weighting_dir = Path.home() / 'Documents/python_projects/weighting'
# weighting_dir.exists()
sys.path.append(str(weighting_dir))  # needed
import src.microweight as mw

from timeit import default_timer as timer


# %% reimports
reload(adv)
reload(fsw)
reload(gwp)
reload(mw)
reload(pc)
reload(rpt)
reload(rwp)
reload(test)
# reload(gwp)


# %% physical locations
WINDATADIR = '/media/don/ignore/data/' # files that were created in Windows version of this
DIR_FOR_OFFICIAL_PUFCSV = r'/media/don/data/puf_files/puf_csv_related_files/PSL/2020-08-20/'

# working storage
SCRATCHDIR = '/media/don/scratch/'
OUTDIR = '/media/don/pufanalysis_output/'

# %%  relative locations to use

# input locations
PUFDIR = DIR_FOR_OFFICIAL_PUFCSV
WEIGHTDIR = DIR_FOR_OFFICIAL_PUFCSV

# output locations
OUTDATADIR = OUTDIR + 'data/'
OUTTABDIR = OUTDIR + 'result_tables/'
OUTWEIGHTDIR = OUTDIR + 'weights/'


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

# get initial national weights, divide by 100, add pid, and save a csv file for each year we will work with
# fsw.save_pufweights(wtpath=WEIGHTS_USE, outdir=OUTWEIGHTDIR, years=(2017, 2018))

# %% 1. Advance puf.csv to a future year, save as parquet file
# add pid and filer, and save as puf+str(year).parquet
# 1.a) advance official puf.csv with official weights, grow factors, and ratios
fsw.advance_and_save_puf(
    year=2017,
    pufpath=PUF_USE,
    growpath=GF_USE,
    wtpath=WEIGHTS_USE,
    ratiopath=RATIOS_USE,
    outdir=OUTDATADIR)

# 1.b) Alternative: advance by "regrowing" with custom growfactors and without puf_ratios
# Not currently implemented

# %% 2. Get potential national targets for 2017, previously created
# from common_stub	incrange	pufvar	irsvar	irs	table_description	column_description	src	excel_column
# should have variables:
#   common_stub	incrange, pufvar, irsvar, irs, table_description, column_description, src	excel_column
ptargets = fsw.get_potential_national_targets(targets_fname=POSSIBLE_TARGETS)  # targs.ptargets, .ptarget_names
# targs._fields
# targs.ptarget_names
# targs.ptargets.columns.str.contains('_nnz')

# %% 3. Create pufsub -- puf subset of filers, with just those variables needed for potential targets
# from puf{year}.parquet file; only includes filer records
# adds pid, filer, stubs, and target variables
pufsub = fsw.prep_puf(OUTDATADIR + 'puf2017.parquet', ptargets)

# % get differences from targets at initial weights and produce report
weights_initial = fsw.get_pufweights(wtpath=WEIGHTS_USE, year=2017)  # adds pid and shortname
pdiff_init = rwp.get_pctdiffs(pufsub, weights_initial, ptargets)
np.round(np.nanquantile(pdiff_init.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_init,
    outfile=OUTTABDIR + 'baseline_national.txt',
    title='2017 puf values using official weights, growfactors, and puf_ratios versus IRS targets.')


# %% 4. Reweight national puf to come closer to targets
drops_national = test.get_drops_national(pdiff_init)
weights_reweight = rwp.puf_reweight(pufsub, weights_initial, ptargets, method='ipopt', drops=drops_national)
# stub 1 gives some trouble

# report on percent differences
pdiff_reweighted = rwp.get_pctdiffs(pufsub, weights_reweight, ptargets)
np.round(np.nanquantile(pdiff_reweighted.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_reweighted,
    outfile=OUTTABDIR + 'reweighted_national.txt',
    title='Reweighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_init)

# %% BREAK: define which states are of interest - MUST stay the same from here forward
# choose or modify ONE of the following
compstates = ['NY', 'AR', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX']
# see pc.STATES, STATES_DCPROA, STATES_DCPROAUS
compstates= pc.STATES[0:20]
compstates = pc.STATES


# %% 5. Prepare state targets and drop combinatoins for the states of interest
# get df with ht2 shares
ht2targets = test.get_potential_state_targets(
    pufsub,
    weightdf=weights_reweight,
    ht2sharespath=HT2_SHARES,  # currently this is from the old Windows run, need to update
    compstates=compstates)

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

drops_states = test.get_drops_states(ht2targets)
ht2wide = test.get_ht2wide_states(ht2targets)


# %% 6. WORK AREA to define target variables
# targvars are the variables we will target -- must be in ht2_possible

targvars = ['nret_all', 'mars1', 'mars2',  # num returns total and by filing status
            'c00100',   # AGI
            'e00200', 'e00200_nnz',  # wages
            'e00300', 'e00300_nnz',  # taxable interest income
            'e00600', 'e00600_nnz',  # ordinary dividends
            'e00900',  # business and professional income
            'e26270',  # partnership/S Corp income
            'c01000',  # capital gains (tc doc says see .py file, though)
            # deductions
            'c17000','c17000_nnz',  # medical expenses deducted
            'c18300', 'c18300_nnz']  # SALT amount deducted

# for testing purposes, here are some useful subsets of targvars
targvars2 = ['nret_all']
targvars2 = ['nret_all', 'c00100']
targvars2 = ['nret_all', 'c00100', 'e00200']
targvars2 = ['nret_all', 'mars1', 'c00100']
targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']
targvars2 = ['nret_all', 'c00100', 'e00200', 'c18300']
# set targvars = one of the above during test runs

# verify that targvars are in the data

len(targvars)
['good' for var in targvars if var in ht2wide.columns]


# %% 7. Construct national weights as sums of unrestricted state weights

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

weights_geosums = gwp.get_geoweight_sums(
    pufsub,
    weightdf=weights_reweight,
    targvars=targvars,
    ht2wide=ht2wide,
    dropsdf_wide=drops_states,
    outpath=OUTWEIGHTDIR + 'allweights2017_geo_unrestricted.csv',
    stubs = None)  # None, or list or tuple of stubs


pdiff_geosums = rwp.get_pctdiffs(pufsub, weights_geosums, ptargets)
np.round(np.nanquantile(pdiff_geosums.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_geosums,
    outfile=OUTTABDIR + 'geosums_national.txt',
    title='Unrestricted geosum weighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_reweighted)


# %% 8. Reweight the national file to come closer to targets
drops_national_geo = test.get_drops_national(pdiff_geosums)
weights_georeweight = rwp.puf_reweight(pufsub, weights_geosums, ptargets, method='ipopt', drops=drops_national_geo)

# report on percent differences
pdiff_georeweighted = rwp.get_pctdiffs(pufsub, weights_georeweight, ptargets)
np.round(np.nanquantile(pdiff_georeweighted.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_georeweighted,
    outfile=OUTTABDIR + 'geosums_reweighted_national.txt',
    title='Geosum reweighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_geosums)


# %% 9. Update state-stub targets to reflect new slightly-revised pufsums
# remember that new pufsums won't be exactly like old so we should recalibrate
# state targets
ht2targets_updated = test.get_potential_state_targets(
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

drops_states_updated = test.get_drops_states(ht2targets_updated)
ht2wide_updated = test.get_ht2wide_states(ht2targets_updated)


# %% 10. Get state weights
opts_newt = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'stepmethod': 'jac',  # jac or jvp for newton; also vjp, findiff if lsq
    'init_beta': 0.5,
    'init_p': 0.75, # 0.75 default,
    # 'maxp_tol': 0.01, # max pct diff tolerance .01 is 1/100 percent
    'max_iter': 20,  # 20 default
    'startup_iter': 8,  # 8 default number of iterations for the startup period
    'startup_p': .25,  # .25 default p, the step multiplier in the startup period
    'linesearch': False,  # True default
    'quiet': True}
# opts.update({'stepmethod': 'jvp'})

opts_lsq = {
    'scaling': True,
    'scale_goal': 1e1,
    'init_beta': 0.5,
    'stepmethod': 'jac',  # jac or jvp for newton; also vjp, findiff if lsq
    'quiet': True}

opts = opts_newt
opts = opts_lsq

tmp = gwp.get_geo_weights_stub(
    pufsub,
    weightdf=weights_georeweight,
    targvars=targvars2,  # use targvars or a variant
    ht2wide=ht2wide_updated,
    dropsdf_wide=drops_states_updated,
    method='poisson-newton',  # poisson-lsq or poisson-newton
    options=opts,
    stub=7)

# compare results to targets for a single stub

targs_used = targvars2
stub = 7

df = pufsub.loc[pufsub['ht2_stub'] ==stub, ['pid', 'ht2_stub'] + targs_used]
htstub = ht2wide_updated.loc[ht2wide_updated['ht2_stub']==stub, ['ht2_stub', 'stgroup'] + targs_used]

sts = htstub.stgroup.tolist()  # be sure that target rows and whs columns are in sts order
xmat = np.asarray(df[targs_used], dtype=float)
targmat = np.asarray(htstub.loc[:, targs_used])
whs = np.asarray(tmp.loc[tmp['ht2_stub']==stub, sts], dtype=float)

targopt = np.dot(whs.T, xmat)
diff = targopt - targmat
pdiff = diff / targmat * 100
sspd = np.square(pdiff).sum()
sspd
np.quantile(pdiff, qtiles)

geomethod = 'qmatrix-ipopt'
opts_q = {'qmax_iter': 50,
           'quiet': True,
           'qshares': None,  # qshares or None
           'xlb': 0.1,
           'xub': 100,
           'crange': .0001,
           'linear_solver': 'ma57'
           }

grouped = pufsub.groupby('ht2_stub')

a = timer()
final_geo_weights = grouped.apply(gwp.get_geo_weights,
                                weightdf=weights_georeweight,
                                targvars=targvars2,
                                ht2wide=ht2wide_updated,
                                dropsdf_wide=drops_states_updated,
                                independent=False,
                                geomethod = geomethod,
                                options=opts_q,
                                intermediate_path=SCRATCHDIR)
b = timer()
b - a
(b - a) / 60





weights_states = gwp.get_geoweight_sums(
    pufsub,
    weightdf=weights_reweight,
    targvars=targvars,
    ht2wide=ht2wide,
    dropsdf_wide=dropsdf_wide,
    outpath=OUTWEIGHTDIR + 'allweights2017_geo_unrestricted.csv',
    stubs = 'all')  # 'all', or list or tuple of stubs


pdiff_geosums = rwp.get_pctdiffs(pufsub, weights_geosums, ptargets)
np.round(np.nanquantile(pdiff_geosums.abspdiff, qtiles), 2)

rpt.wtdpuf_national_comp_report(
    pdiff_geosums,
    outfile=OUTTABDIR + 'geosums_national.txt',
    title='Unrestricted geosum weighted 2017 puf values versus IRS targets.',
    ipdiff_df=pdiff_reweighted)

opts.update({'stepmethod': 'jac', 'x_scale': 'jac'})
opts.update({'stepmethod': 'jvp', 'x_scale': 'jac'})
opts.update({'stepmethod': 'vjp', 'x_scale': 'jac'})
opts.update({'stepmethod': 'jvp-linop', 'x_scale': 1.0})  # may not work well on real problems
opts.update({'stepmethod': 'findiff', 'x_scale': 'jac'})
opts.update({'x_scale': 'jac'})
opts.update({'x_scale': 1.0})
opts.update({'max_nfev': 200})

options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!
    'init_beta': 0.5,
    'stepmethod': 'jvp',  # vjp, jvp, full, finite-diff
    'max_nfev': 100,
    'ftol': 1e-7,
    'x_scale': 'jac',
    'quiet': True}



options_defaults = {
    'scaling': True,
    'scale_goal': 10.0,  # this is an important parameter!!
    'init_beta': 0.5,
    'stepmethod': 'jac',  # jvp or jac, jac seems to work better
    'max_iter': 20,
    'linesearch': True,
    'init_p': 0.75,  # less than 1 seems important
    'maxp_tol': .01,  # .01 is 1/100 of 1% for the max % difference from target
    'startup_period': True,  # should we have a separate startup period?
    'startup_imaxpdiff': 1e6,  # if initial maxpdiff is greater than this go into startup mode
    'startup_iter': 8,  # number of iterations for the startup period
    'startup_p': .25,  # p, the step multiplier in the startup period
    'quiet': True}




# djb stopped here 6/4/2021

# %% END


# %%
# %% OLD below here
# %% PLAY AREA (To be removed)

# %% create report on results with the geo revised national weights


# %% construct new state targets for geoweighting geoweight: get revised national weights based on independent state weights

# get NATIONAL weights to use as starting point for ht2 stubs -- from step above
wfname = OUTDATADIR + 'weights2017_georwt1.csv'
weights = pd.read_csv(wfname)

# get NATIONAL pufsums with these weights, by ht2 stub
# these are the amounts we will share across states
pufsums_ht2 = rwp.get_wtdsums(pufsub, ptarget_names, weights, stubvar='ht2_stub')
pufsums_ht2.columns
pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')

# collapse ht2 shares to the states we want (should be the same as before)
ht2_collapsed = gwp.collapse_ht2(HT2_SHARES, compstates)
pu.uvals(ht2_collapsed.pufvar)  # the variables available for targeting

# create targets by state and ht2_stub from pufsums and collapsed shares
ht2_collapsed

# merge in the puf sums with the new (final) NATIONAL weights so that we can
# share the NATIONAL sums to the states, to be our state targets
ht2targets = pd.merge(ht2_collapsed, pufsums_ht2long, on=['pufvar', 'ht2_stub'])
ht2targets.info()
ht2targets['target'] = ht2targets.pufsum * ht2targets.share
ht2targets['diff'] = ht2targets.target - ht2targets.ht2
ht2targets['pdiff'] = ht2targets['diff'] / ht2targets.ht2 * 100
ht2targets['abspdiff'] = np.abs(ht2targets['pdiff'])
ht2targets.to_csv(SCRATCHDIR + 'ht2targets_temp.csv', index=None)  # temporary


# %% explore the updated state-variable-stub targets
check = ht2targets.sort_values(by='abspdiff', axis=0, ascending=False)
np.nanquantile(check.abspdiff, qtiles)

# let's look at some targets vs their HT2 values
# what's true of one state is true of all
var = "c04800"
st = "AR"
temp = check.query('pufvar == @var & stgroup==@st ').sort_values(by='ht2_stub', axis=0, ascending=True)


# %% create a wide boolean dataframe indicating whether a target will be dropped
qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values
dropsdf = ht2targets.query(qxnan)[['stgroup', 'ht2_stub', 'pufvar']]
dropsdf['drop'] = True
dropsdf_stubs = ht2_collapsed.query('ht2_stub > 0')[['stgroup', 'ht2_stub', 'pufvar']]
dropsdf_full = pd.merge(dropsdf_stubs, dropsdf, how='left', on=['stgroup', 'ht2_stub', 'pufvar'])
dropsdf_full.fillna(False, inplace=True)
dropsdf_wide = dropsdf_full.pivot(index=['stgroup', 'ht2_stub'], columns='pufvar', values='drop').reset_index()

keepvars = ['stgroup', 'ht2_stub', 'pufvar', 'target']
ht2wide = ht2targets[keepvars].pivot(index=['stgroup', 'ht2_stub'], columns='pufvar', values='target').reset_index()

ht2_vars = pu.uvals(ht2_collapsed.pufvar)
ptarget_names  # possible targets, if in ht2
ht2_possible = [var for var in ptarget_names if var in ht2_vars]

pufsub.columns
pufsub[['ht2_stub', 'nret_all']].groupby(['ht2_stub']).agg(['count'])

targvars = ['nret_all', 'mars1', 'mars2',  # num returns total and by filing status
            'c00100',   # AGI
            'e00200', 'e00200_nnz',  # wages
            'e00300', 'e00300_nnz',  # taxable interest income
            'e00600', 'e00600_nnz',  # ordinary dividends
            'e00900',  # business and professional income
            'e26270',  # partnership/S Corp income
            'c01000',  # capital gains (tc doc says see .py file, though)
            # deductions
            'c17000','c17000_nnz',  # medical expenses deducted
            'c18300', 'c18300_nnz']  # SALT amount deducted
['good' for var in targvars if var in ht2_possible]

targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']


# %% Use independent weights as starting point
wfname = OUTDATADIR + 'allweights2017_geo_unrestricted.csv'
qshares = pd.read_csv(wfname)
qshares.info()

# stvars = [s for s in qshares.columns if s not in ['pid', 'ht2_stub', 'weight', 'geoweight_sum']]
stvars = compstates + ['other']
qshares = qshares[['pid', 'ht2_stub'] + stvars]
qshares[stvars] = qshares[stvars].div(qshares[stvars].sum(axis=1), axis=0)

# we will form the initial Q matrix for each ht2_stub from qshares

# check
qshares[stvars].sum(axis=1).sum()

# %% save before running final loop
# targvars
# ht2wide
# pufsub
# dropsdf_wide

save_list = [targvars, ht2wide, pufsub, dropsdf_wide]
save_name = SCRATCHDIR + 'pickle.pkl'

open_file = open(save_name, "wb")
pickle.dump(save_list, open_file)
open_file.close()


# %% run the final loop
#geomethod = 'qmatrix-ipopt'  # qmatrix-ipopt or qmatrix-lsq
#reweight_method = 'ipopt'  # ipopt or lsq
#wfname_national = PUFDIR + 'weights_georwt1_' + geomethod + '_' + reweight_method + '.csv'
wfname_national = WEIGHTDIR + 'weights2017_georwt1.csv'
wfname_national
final_national_weights = pd.read_csv(wfname_national)
# final_national_weights.head(20)

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

# geomethod = 'qmatrix-lsq'
# options = {'qmax_iter': 50,
#            'qshares': qshares,
#            'verbose': 0,
#            'xlb': 0.2,
#            'scaling': True,
#            'method': 'bvls',  # bvls (default) or trf - bvls usually faster, better
#            'lsmr_tol': 'auto' # auto'  # 'auto'  # 'auto' or None
#            }


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

# geo_weights[list(compstates) + ['other']].sum(axis=1)
# note that the method here is lsq
# wfname = PUFDIR + 'weights_geo_rwt.csv'
# nat_geo_weights[['pid', 'geo_rwt']].to_csv(wfname, index=None)
final_geo_name = WEIGHTDIR + 'allweights2017_geo_restricted.csv'
final_geo_name
final_geo_weights.to_csv(final_geo_name, index=None)


# %% create report on results with the state weights
date_id = date.today().strftime("%Y-%m-%d")

ht2_compare = pd.read_csv(IGNOREDIR + 'ht2targets_temp.csv')  # temporary
pu.uvals(ht2_compare.pufvar)
sweights = pd.read_csv(WEIGHTDIR + 'allweights2017_geo_restricted.csv')

asl = fht.get_allstates_wsums(pufsub, sweights)
pu.uvals(asl.pufvar)
comp = fht.get_compfile(asl, ht2_compare)
pu.uvals(comp.pufvar)

outfile = TABDIR + 'comparison_state_values_vs_state_puf-based targets_' + date_id +'.txt'
title = 'Comparison of state results to puf-based state targets'
fht.comp_report(comp, outfile, title)



# %% create file with multiple national weights
# basenames of weight csv files
weight_filenames = [
    'weights2017_default',  # weights from tax-calculator
    'weights2017_reweight1',  # weights_regrown reweighted to national totals, with ipopt -- probably final
    'weights2017_geo_unrestricted',  # sum of state weights, developed using weights_rwt1_ipopt as starting point
    'weights2017_georwt1',  # the above sum of state weights reweighted to national totals, with ipopt
    # the weights immediately above are the weights apportioned to states
    ]

# in addition, there are two files of national weights that have been apportioned to states
#   that have state weights as well as the national weight, with an "all" prefix:
#     allweights_geo_unrestricted_qmatrix-ipopt -- counterpart to geo_unrestricted_qmatrix-ipopt
#     allweights_geo_restricted_qmatrix-ipopt -- counterpart to weights_georwt1_qmatrix-ipopt_ipopt
#       this latter file has the final state weights

def f(fnames, dir):
    fullfnames = [s + '.csv' for s in fnames]  # keep this separately
    paths = [dir + s for s in fullfnames]

    pdlist = []
    for p, f in zip(paths, fullfnames):
        # CAUTION: ASSUMES the file has only 2 columns, pid and a weight
        df = pd.read_csv(p)
        df['file_source'] = f
        pdlist.append(df)

    weights_stacked = pd.concat(pdlist)
    return weights_stacked

national_weights = f(weight_filenames, WEIGHTDIR)
national_weights.to_csv(WEIGHTDIR + 'national_weights_stacked.csv', index=None)

# weight_df = rwp.merge_weights(weight_filenames, PUFDIR)  # they all must be in the same directory

# weight_df.to_csv(PUFDIR + 'all_weights.csv', index=None)
# weight_df.sum()

# take a look
nat_wide = national_weights.drop(columns='file_source').pivot(index=['pid'], columns='shortname', values='weight').reset_index()
col_order = ['pid', 'weights2017_default', 'reweight1', 'geoweight_sum', 'georeweight1']
nat_wide = nat_wide[col_order]
nat_wide.sort_values(by='pid', inplace=True)
nat_wide.to_csv(WEIGHTDIR + 'national_weights_wide.csv', index=None)


# %% get weights for 2018 and save puf2018_weighted
 # Create a base 2018 puf as follows:
 #     - start with the previously created puf for 2018, which is simply the
 #       puf2017_regrown extrapolated to 2018 with default growfactors, with
 #       taxdata weights for 2018, and WITHOUT adjust_ratios applied
 #     - get the best set of national weights we have for 2017 grow them
 #       based on how the sum of default weights grows
 #     - use these weights where we have them; where not, use default weights

default_weights = pd.read_csv(DIR_FOR_OFFICIAL_PUF + 'puf_weights.csv')
dw2017 = default_weights.sum().WT2017
dw2018 = default_weights.sum().WT2018
wtgrowfactor = dw2018 / dw2017

# get best national weights
# 'weights_georwt1_qmatrix-ipopt_ipopt'

sweights2017 = pd.read_csv(WEIGHTDIR + 'allweights2017_geo_restricted.csv')

puf2018 = pd.read_parquet(TCOUTDIR + 'puf2018.parquet', engine='pyarrow')

puf2018_weighted = puf2018.copy().rename(columns={'s006': 's006_default'})
puf2018_weighted = pd.merge(puf2018_weighted,
                            sweights2017.loc[:, ['pid', 'weight']],
                            how='left',
                            on='pid')
puf2018_weighted['weight2018_2017filers'] = puf2018_weighted.weight * wtgrowfactor
puf2018_weighted['s006'] = puf2018_weighted.weight2018_2017filers
puf2018_weighted.s006.fillna(puf2018_weighted.s006_default, inplace=True)
puf2018_weighted[['pid', 's006_default', 'weight', 'weight2018_2017filers', 's006']].head(20)

puf2018_weighted.drop(columns=['weight', 'weight2018_2017filers'], inplace=True)
pu.uvals(puf2018_weighted.columns)

puf2018_weighted.to_parquet(TCOUTDIR + 'puf2018_weighted' + '.parquet', engine='pyarrow')


# finally, create state weights for 2018 using the shares we have for 2017
# we know this isn't really right, but shouldn't be too bad (for now) for a single year
# this should be as simple as multiplying all weights by wtgrowfactor
wtvars = [s for s in sweights2017.columns if s not in ['pid', 'ht2_stub']]
sweights2018 = sweights2017.copy()
sweights2018[wtvars] = sweights2018[wtvars] * wtgrowfactor

sweights2018.to_csv(WEIGHTDIR + 'allweights2018_geo2017_grown.csv', index=None)


# %% djb experiment 2021: poisson



# %% OLD BELOW HERE -- VARIANT: drops for lsq: define any variable-stub combinations to drop via a drops dataframe
# create drops data frame for when we use lsq for targeting instead of ipopt

goodvars = ['nret_all', 'mars1', 'mars2', 'mars4',
            'c00100',
            'e00200', 'e00200_nnz',
            'e00300', 'e00300_nnz',
            'e00600', 'e00600_nnz',
            'c01000pos', 'c01000pos_nnz',
            'c01000neg', 'c01000neg_nnz',
            'e01500', 'e01500_nnz',
            'c02500', 'c02500_nnz',
            'e26270pos', 'e26270pos_nnz',
            'e26270neg', 'e26270neg_nnz',
            # 'c17000',
            'c18300', 'c18300_nnz',
            'c19200',  # 'c19200_nnz',
            'c19700'  # , 'c19700_nnz'
            ]

qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values
qx1 = '(pufvar not in @goodvars)'

bad_stub1_4_vars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']
qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @bad_stub1_4_vars)"

qx = qxnan + " or " + qx1 + " or " + qx2
qx

drops_lsq = pdiff_init.query(qx).copy()
drops_lsq
