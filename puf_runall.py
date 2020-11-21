# https://stackoverflow.com/questions/21868369/pycharm-hanging-for-a-long-time-in-ipython-console-with-big-data
# to (maybe) fix pycharm hanging:
# Files -> Settings -> Build, Execution, Deployment -> Python Debugger
# switch on the "Gevent Compatible" flag

# %% imports
import sys
import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import functions_advance_puf as adv
import functions_reweight_puf as rwp
import functions_geoweight_puf as gwp

import puf_constants as pc
import puf_utilities as pu

# microweight - apparently we have to tell python where to find this
sys.path.append('c:/programs_python/weighting/')  # needed
import src.microweight as mw

from timeit import default_timer as timer
from importlib import reload
# reload(pc)
# reload(rwp)


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'


# %% paths to specific already existing files
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'

# growfactors
GF_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'growfactors.csv'
GF_CUSTOM = DATADIR + 'growfactors_custom.csv'  # selected growfactors reflect IRS growth between 2011 and 2017
GF_ONES = DATADIR + 'growfactors_ones.csv'

WEIGHTS_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'puf_weights.csv'

POSSIBLE_TARGETS = DATADIR + 'targets2017_possible.csv'

HT2_SHARES = DATADIR + 'ht2_shares.csv'


# %% names of files to create
PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)
compstates = ['NY', 'AR', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX']


# %% ONETIME: create and save default and regrown 2017 pufs, and add filer indicator
puf = pd.read_csv(LATEST_OFFICIAL_PUF)

adv.advance_puf(puf, 2017, PUF_DEFAULT)

adv.advance_puf_custom(puf, 2017,
                       gfcustom=GF_CUSTOM,
                       gfones=GF_ONES,
                       weights=WEIGHTS_OFFICIAL,
                       savepath=PUF_REGROWN)


# %% define possible targets - we may not use all of them
ptargets = rwp.get_possible_targets(targets_fname=POSSIBLE_TARGETS)
ptargets
ptargets.info()
ptarget_names = ptargets.columns.tolist()
ptarget_names.remove('common_stub')
ptarget_names


# %% prepare a version of the puf for reweighting
# do the following:
#   add stub variables
#   create mars1, mars2, ... marital status indicators
#   create any positive or negative variables needed
#   create any needed nnz indicators
#   keep only the needed variables pid, common_stub,

pufrg = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
pufrg.info()
pu.uvals(pufrg.columns)
pufrg.filer.sum()

pufsub = rwp.prep_puf(pufrg, ptargets)
pufsub.info()
pu.uvals(pufsub.columns)


# %% get weights for regrown file
# all weight files will have pid, weight, shortname as columns
weights_regrown = pd.read_csv(PUFDIR + 'weights_regrown.csv')
weights_regrown  # MUST have columns pid, weight -- no other columns or names
# weights_regrown.iloc[:, [0, 1]]


# %% get % differences from targets at initial weights
pdiff_init = rwp.get_pctdiffs(pufsub, weights_regrown, ptargets)
pdiff_init.shape
np.nanquantile(pdiff_init.abspdiff, qtiles)
np.nanquantile(pdiff_init.pdiff, qtiles)
pdiff_init.head(15)
pdiff_init.query('abspdiff > 10')

pu.uvals(pdiff_init.pufvar)


# %% ipopt: define any variable-stub combinations to drop via a drops dataframe

# variables we don't want to target (e.g., taxable income or tax after credits)
untargeted = ['c01000', 'c01000_nnz',  # we are targeting the pos and neg versions
              'c04800', 'c04800_nnz',
              'c09200', 'c09200_nnz',
              'taxac_irs', 'taxac_irs_nnz']

badvars = ['c02400', 'c02400_nnz']  # would like to target but values are bad
bad_stub1_4_vars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']

# define query
qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values
qx0 = "(pufvar in @untargeted)"
qx1 = "(pufvar in @badvars)"
qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @bad_stub1_4_vars)"
qx = qxnan + " or " + qx0 + " or " + qx1 + " or " + qx2
qx

drops_ipopt = pdiff_init.query(qx).copy()
drops_ipopt.sort_values(by=['common_stub', 'pufvar'], inplace=True)
drops_ipopt


# %% lsq: define any variable-stub combinations to drop via a drops dataframe

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


# %% reweight the puf file
method = 'ipopt'  # ipopt or lsq
drops = drops_ipopt  # use ipopt or lsq

# method = 'lsq'  # ipopt or lsq
# drops = drops_lsq  # use ipopt or lsq

# temp = pufsub.query('common_stub==2')  # this stub is the hardest for both solvers

a = timer()
new_weights = rwp.puf_reweight(pufsub, weights_regrown, ptargets, method=method, drops=drops)
b = timer()
b - a
# new_weights.sum()

weights_save = new_weights.copy()
weights_save['shortname'] = 'reweight1'
weights_save = weights_save.drop(columns='weight').rename(columns={'reweight': 'weight'})

wfname = PUFDIR + 'weights_reweight1_' + method + '.csv'
weights_save.to_csv(wfname, index=None)


# %% check pdiffs
# pd.set_option('display.max_columns', 7)
pdiff_rwt = rwp.get_pctdiffs(pufsub, new_weights[['pid', 'reweight']], ptargets)
pdiff_rwt.shape
pdiff_rwt.head(20)[['common_stub', 'pufvar', 'pdiff', 'abspdiff']]
pdiff_rwt.query('abspdiff > 10')
pu.uvals(pdiff_rwt.pufvar)


# %% create report on results from the reweighting
# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interst

date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
wfname = PUFDIR + 'weights_reweight1_ipopt.csv'
weights_comp = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_reweighted_ipopt_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf, ipopt method, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_reweight=weights_comp,  # new_weights[['pid', 'reweight']],
                 weights_init=weights_regrown,
                 compvars=ptargets,
                 dropvars=None,
                 outfile=rfname, title=rtitle)

pu.uvals(pufsub.columns)
pu.uvals(ptargets.columns)


# %% geoweight: get national weights by adding up unrestricted state weights
# get weights to use as starting point for ht2 stubs
wfname = PUFDIR + 'weights_reweight1_ipopt.csv'
weights_national = pd.read_csv(wfname)

# get national pufsums with these weights, for ht2 stubs
# these are the amounts we will share across states
pufsums_ht2 = rwp.get_wtdsums(pufsub, ptarget_names, weights_national, stubvar='ht2_stub')
pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')
pu.uvals(pufsums_ht2long.pufvar)

# collapse ht2 shares to the states we want
ht2_collapsed = gwp.collapse_ht2(HT2_SHARES, compstates)
pu.uvals(ht2_collapsed.pufvar)
pu.uvals(ht2_collapsed.ht2var)

# create targets by state and ht2_stub from pufsums and collapsed shares
ht2_collapsed
ht2targets = pd.merge(ht2_collapsed, pufsums_ht2long, on=['pufvar', 'ht2_stub'])
ht2targets.info()
ht2targets['target'] = ht2targets.pufsum * ht2targets.share
ht2targets['diff'] = ht2targets.target - ht2targets.ht2
ht2targets['pdiff'] = ht2targets['diff'] / ht2targets.ht2 * 100
ht2targets['abspdiff'] = np.abs(ht2targets['pdiff'])

# explore the result
check = ht2targets.sort_values(by='abspdiff', axis=0, ascending=False)
np.nanquantile(check.abspdiff, qtiles)


# %% define HT2 targets to drop
# create a wide boolean dataframe indicating whether a target will be dropped
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

targvars = ['nret_all', 'mars1', 'mars2', 'c00100', 'e00200', 'e00200_nnz',
            'e00300', 'e00300_nnz', 'e00600', 'e00600_nnz',
            'c01000',
            # deductions
            'c17000','c17000_nnz',
            'c18300', 'c18300_nnz']
['good' for var in targvars if var in ht2_possible]

targvars2 = ['nret_all']
targvars2 = ['nret_all', 'c00100']
targvars2 = ['nret_all', 'c00100', 'e00200']
targvars2 = ['nret_all', 'mars1', 'c00100']
targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']
targvars2 = ['nret_all', 'c00100', 'e00200', 'c18300']


# %% common options for geoweighting

uo = {'qmax_iter': 10}
uo = {'qmax_iter': 1, 'independent': True}
uo = {'qmax_iter': 10, 'quiet': True}
uo = {'qmax_iter': 3, 'quiet': True, 'verbose': 2}

# raking options (there aren't really any)
uo = {'qmax_iter': 10}

# empcal options
uo = {'qmax_iter': 10, 'objective': 'ENTROPY'}
uo = {'qmax_iter': 10, 'objective': 'QUADRATIC'}

# ipopt options
uo = {'qmax_iter': 10,
      'quiet': True,
      'xlb': 0.1,
      'xub': 100,
      'crange': .0001
      }

# lsq options
uo = {'qmax_iter': 10,
      'verbose': 0,
      'xlb': 0.3,
      'scaling': False,
      'method': 'bvls',  # bvls (default) or trf - bvls usually faster, better
      'lsmr_tol': 'auto'  # 'auto'  # 'auto' or None
      }


# %% get new national weights by getting weights for each state (for each record) and summing them
wfname_init = PUFDIR + 'weights_reweight1_ipopt.csv'
weights_init = pd.read_csv(wfname_init)

grouped = pufsub.groupby('ht2_stub')
# targvars, ht2wide, dropsdf_wide, independent=False

# choose one of the following combinations of geomethod and options
geomethod = 'qmatrix'  # does not work well
options = {}

geomethod = 'qmatrix-ipopt'
options = {'quiet': True,
           'xlb': 0.1,
           'xub': 100,
           'crange': .0001
           }

geomethod = 'qmatrix-lsq'
options = {'verbose': 0,
           'xlb': 0.2,
           'scaling': False,
           'method': 'bvls',  # bvls (default) or trf - bvls usually faster, better
           'lsmr_tol': 'auto'  # 'auto'  # 'auto' or None
           }

a = timer()
nat_geo_weights = grouped.apply(gwp.get_geo_weights,
                                weightdf=weights_init,
                                targvars=targvars,
                                ht2wide=ht2wide,
                                dropsdf_wide=dropsdf_wide,
                                independent=True,
                                geomethod=geomethod,
                                options=options)
b = timer()
b - a


# save just the pid and national weights
wfname_result = PUFDIR + 'weights_geo_unrestricted_' + geomethod + '.csv'
weights_save = nat_geo_weights.copy()
weights_save = weights_save.loc[:, ['pid', 'geoweight_sum']].rename(columns={'geoweight_sum': 'weight'})
weights_save['shortname'] = 'geoweight_sum'
weights_save.to_csv(wfname_result, index=None)

# write the full file of state weights to disk
nat_geo_weights.to_csv(PUFDIR + 'allweights_geo_unrestricted_' + geomethod + '.csv', index=None)

nat_geo_weights.sum()

g = nat_geo_weights.geoweight_sum / nat_geo_weights.weight
np.quantile(g, qtiles)


# %% create report on results with the geo revised national weights

# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interest
wfname_base = PUFDIR + 'weights_reweight1_ipopt.csv'
weights_base = pd.read_csv(wfname_base)

# method = 'ipopt'  # ipopt or lsq
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
# choose a geomethod
geomethod = 'qmatrix-ipopt'  # qmatrix-ipopt or qmatrix-lsq
wfname = PUFDIR + 'weights_geo_unrestricted_' + geomethod + '.csv'
weights_comp = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_geo_reweighted_' + geomethod + '_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf then georeweighted compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_reweight=weights_comp,  # new_weights[['pid', 'reweight']],
                 weights_init=weights_base,
                 compvars=ptargets,
                 dropvars=None,
                 outfile=rfname, title=rtitle)


# %% reweight the geo revised national weights
# in theory, these will be the weights we use to create the national file
# that will be shared out to states

geomethod = 'qmatrix-ipopt'  # qmatrix-ipopt or qmatrix-lsq
wfname_init = PUFDIR + 'weights_geo_unrestricted_' + geomethod + '.csv'
weights_init = pd.read_csv(wfname_init).rename(columns={'geo_rwt': 'weight'})

# choose a reweight method and a corresponding set of drops
reweight_method = 'ipopt'  # ipopt or lsq
drops = drops_ipopt

a = timer()
new_weights = rwp.puf_reweight(pufsub, weights_init, ptargets, method=reweight_method, drops=drops)
b = timer()
b - a

# wtname_result = 'georwt1_' + geomethod + '_' + reweight_method
# wfname_result = PUFDIR + 'weights_georwt1_' + geomethod + '_' + reweight_method + '.csv'
# new_weights[['pid', 'reweight']].rename(columns={'reweight': wtname_result}).to_csv(wfname_result, index=None)

weights_save = new_weights.copy()
weights_save['shortname'] = 'georeweight1'
weights_save = weights_save.drop(columns='weight').rename(columns={'reweight': 'weight'})

wfname = PUFDIR + 'weights_georwt1_' + method + '.csv'
weights_save.to_csv(wfname, index=None)


# %% create report on results with the revised georevised national weights
# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interest
wfname_base = PUFDIR + 'weights_geo_unrestricted_qmatrix-ipopt.csv'
weights_base = pd.read_csv(wfname_base)

# method = 'ipopt'  # ipopt or lsq
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
# wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
weights_comp = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_georeweighted_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf georeweighted ipopt, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_reweight=weights_comp,  # new_weights[['pid', 'reweight']],
                 weights_init=weights_base,
                 compvars=ptargets,
                 dropvars=None,
                 outfile=rfname, title=rtitle)


# %% construct new targets for geoweighting geoweight: get revised national weights based on independent state weights
# get weights to use as starting point for ht2 stubs
wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
weights = pd.read_csv(wfname)

# get national pufsums with these weights, for ht2 stubs
# these are the amounts we will share across states
pufsums_ht2 = rwp.get_wtdsums(pufsub, ptarget_names, weights, stubvar='ht2_stub')
pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')

# collapse ht2 shares to the states we want
ht2_collapsed = gwp.collapse_ht2(HT2_SHARES, compstates)

# create targets by state and ht2_stub from pufsums and collapsed shares
ht2_collapsed
ht2targets = pd.merge(ht2_collapsed, pufsums_ht2long, on=['pufvar', 'ht2_stub'])
ht2targets.info()
ht2targets['target'] = ht2targets.pufsum * ht2targets.share
ht2targets['diff'] = ht2targets.target - ht2targets.ht2
ht2targets['pdiff'] = ht2targets['diff'] / ht2targets.ht2 * 100
ht2targets['abspdiff'] = np.abs(ht2targets['pdiff'])

# explore the result
check = ht2targets.sort_values(by='abspdiff', axis=0, ascending=False)
np.nanquantile(check.abspdiff, qtiles)


# create a wide boolean dataframe indicating whether a target will be dropped
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

targvars = ['nret_all', 'mars1', 'mars2', 'c00100', 'e00200', 'e00200_nnz',
            'e00300', 'e00300_nnz', 'e00600', 'e00600_nnz',
            'c01000',
            # deductions
            'c17000','c17000_nnz',
            'c18300', 'c18300_nnz']
['good' for var in targvars if var in ht2_possible]

targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']


# %% Use independent weights as starting point
wfname = PUFDIR + 'allweights_geo_unrestricted_qmatrix-ipopt.csv'
qshares = pd.read_csv(wfname)
qshares.info()

# stvars = [s for s in qshares.columns if s not in ['pid', 'ht2_stub', 'weight', 'geoweight_sum']]
stvars = compstates + ['other']
qshares = qshares[['pid', 'ht2_stub'] + stvars]
qshares[stvars] = qshares[stvars].div(qshares[stvars].sum(axis=1), axis=0)

# we will form the initial Q matrix for each ht2_stub from qshares

# check
qshares[stvars].sum(axis=1).sum()


# %% run the final loop
#geomethod = 'qmatrix-ipopt'  # qmatrix-ipopt or qmatrix-lsq
#reweight_method = 'ipopt'  # ipopt or lsq
#wfname_national = PUFDIR + 'weights_georwt1_' + geomethod + '_' + reweight_method + '.csv'
wfname_national = PUFDIR + 'weights_georwt1_ipopt.csv'
wfname_national
final_national_weights = pd.read_csv(wfname_national)

grouped = pufsub.groupby('ht2_stub')
# targvars, ht2wide, dropsdf_wide, independent=False

# choose one of the following combinations of geomethod and options
geomethod = 'qmatrix'  # does not work well
options = {'qshares': qshares }  # qshares or None

# ipopt took 31 mins
geomethod = 'qmatrix-ipopt'
options = {'qmax_iter': 30,
           'quiet': True,
           'qshares': qshares,  # qshares or None
           'xlb': 0.1,
           'xub': 100,
           'crange': .0001
           }

geomethod = 'qmatrix-lsq'
options = {'qmax_iter': 10,
           'qshares': None,
           'verbose': 0,
           'xlb': 0.2,
           'scaling': False,
           'method': 'bvls',  # bvls (default) or trf - bvls usually faster, better
           'lsmr_tol': 'auto'  # 'auto'  # 'auto' or None
           }

a = timer()
final_geo_weights = grouped.apply(gwp.get_geo_weights,
                                weightdf=final_national_weights,
                                targvars=targvars,
                                ht2wide=ht2wide,
                                dropsdf_wide=dropsdf_wide,
                                independent=False,
                                geomethod=geomethod,
                                options=options)
b = timer()
b - a
(b - a) / 60

final_geo_weights.sum()

# geo_weights[list(compstates) + ['other']].sum(axis=1)
# note that the method here is lsq
# wfname = PUFDIR + 'weights_geo_rwt.csv'
# nat_geo_weights[['pid', 'geo_rwt']].to_csv(wfname, index=None)
final_geo_name = PUFDIR + 'allweights_geo_restricted_' + geomethod + '.csv'
final_geo_name
final_geo_weights.to_csv(final_geo_name, index=None)


# %% look
wipopt = pd.read_csv(PUFDIR + 'weights_geo_restricted.csv')
wrake = pd.read_csv(PUFDIR + 'weights_geo_restricted_raking.csv')
wipopt.sum()
wrake.sum()

wipopt.sum()- wrake.sum()
ht2wide.query('ht2_stub==0')[['stgroup', 'nret_all']]

# raking seems closer


# %% create report on results with the state weights



# %% create file with multiple national weights
# basenames of weight csv files
weight_filenames = [
    'weights_default',  # weights from tax-calculator
    'weights_regrown',  # currently the same as weights_default
    'weights_rwt1_ipopt',  # weights_regrown reweighted to national totals, with ipopt -- probably final
    'weights_geo_unrestricted_qmatrix-ipopt',  # sum of state weights, developed using weights_rwt1_ipopt as starting point
    'weights_georwt1_qmatrix-ipopt_ipopt',  # the above sum of state weights reweighted to national totals, with ipopt
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
        weight_name = df.columns[1]
        df.columns = ['pid', 'weight']
        df['weight_name'] = weight_name
        df['file_source'] = f
        pdlist.append(df)

    weights_stacked = pd.concat(pdlist)
    return weights_stacked

national_weights = f(weight_filenames, PUFDIR)
national_weights.to_csv(PUFDIR + 'national_weights_stacked.csv', index=None)

weight_df = rwp.merge_weights(weight_filenames, PUFDIR)  # they all must be in the same directory

weight_df.to_csv(PUFDIR + 'all_weights.csv', index=None)
weight_df.sum()


# %% DONOTRUN: state weights using the initial reweights
# wfname1 = PUFDIR + 'weights_rwt1_ipopt.csv'
# weights1 = pd.read_csv(wfname1)

# # for testing
# # temp = pufsub.query('ht2_stub in [1, 2]').copy()
# # grouped = temp.groupby('ht2_stub')

# # for real
# grouped = pufsub.groupby('ht2_stub')

# # uo = {'Q': Q_init, 'drops': drops, 'independent': True, 'max_iter': 10}
# a = timer()
# geo_weights = grouped.apply(gwp.get_geo_weights, weights1, targvars2, ht2wide, dropsdf_wide, independent=False)
# b = timer()
# b - a


# geo_weights[list(compstates) + ['other']].sum(axis=1)
