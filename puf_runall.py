
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


# %% ONETIME: create and save default and regrown 2017 pufs
puf = pd.read_csv(LATEST_OFFICIAL_PUF)

adv.advance_puf(puf, 2017, PUF_DEFAULT)

adv.advance_puf_custom(puf, 2017,
                       gfcustom=GF_CUSTOM,
                       gfones=GF_ONES,
                       weights=WEIGHTS_OFFICIAL,
                       savepath=PUF_REGROWN)


# %% ONETIME: save weights from these files
puf_weights = pd.read_parquet(PUF_DEFAULT, engine='pyarrow')[['pid', 's006']].rename(columns={'s006': 's006_default'})
puf_weights.to_csv(PUFDIR + 'weights_default.csv', index=None)

pufrg_weights = pd.read_parquet(PUF_REGROWN, engine='pyarrow')[['pid', 's006']].rename(columns={'s006': 's006_regrown'})
pufrg_weights.to_csv(PUFDIR + 'weights_regrown.csv', index=None)


# %% define possible targets
ptargets = rwp.get_possible_targets(POSSIBLE_TARGETS)
ptargets
ptargets.info()
ptarget_names = ptargets.columns.tolist()
ptarget_names.remove('common_stub')
ptarget_names


# %% prepare a version of the puf for reweighting
# do the following:
#   add filer and stub variables
#   create mars1, mars2, ... marital status indicators
#   create any positive or negative variables needed
#   create any needed nnz indicators
#   keep only the needed variables pid, common_stub,

pufrg = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
pufsub = rwp.prep_puf(pufrg, ptargets)
pufsub.info()
pufsub.columns


# %% get initial weights
init_weights = pd.read_csv(PUFDIR + 'weights_regrown.csv').rename(columns={'s006_regrown': 'weight'})
init_weights  # MUST have columns pid, weight -- no other columns or names


# %% get % differences from targets at initial weights
pdiff_init = rwp.get_pctdiffs(pufsub, init_weights, ptargets)
pdiff_init.shape
np.nanquantile(pdiff_init.abspdiff, qtiles)
np.nanquantile(pdiff_init.pdiff, qtiles)
pdiff_init.head(15)
pdiff_init.query('abspdiff > 10')


# %% ipopt: define any variable-stub combinations to drop via a drops dataframe
badvars = ['c02400', 'c02400_nnz']
bad_stub1_4_vars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']
qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values
qx1 = "(pufvar in @badvars)"
qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @bad_stub1_4_vars)"
qx = qxnan + " or " + qx1 + " or " + qx2
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

a = timer()
new_weights = rwp.puf_reweight(pufsub, init_weights, ptargets, method=method, drops=drops)
b = timer()
b - a

wtname = 'rwt1_' +  method
wfname = PUFDIR + 'weights_rwt1_' + method + '.csv'
new_weights[['pid', 'reweight']].rename(columns={'reweight': wtname}).to_csv(wfname, index=None)

# new_weights.sum()


# %% check pdiffs
pd.set_option('display.max_columns', 7)
pdiff_rwt = rwp.get_pctdiffs(pufsub, new_weights[['pid', 'reweight']], ptargets)
pdiff_rwt.shape
pdiff_rwt.head(20)[['common_stub', 'pufvar', 'pdiff', 'abspdiff']]
pdiff_rwt.query('abspdiff > 10')


# %% create report on results from the reweighting
# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interst

method = 'ipopt'  # ipopt or lsq
# method = 'lsq'
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
wfname = PUFDIR + 'weights_rwt1_' + method + '.csv'
comp_weights = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_reweighted_' + method + '_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf, ' + method + ' method, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_rwt=comp_weights,  # new_weights[['pid', 'reweight']],
                 weights_init=init_weights,
                 targets=ptargets, outfile=rfname, title=rtitle)


# %% geoweight: get revised national weights based on independent state weights
# get weights to use as starting point for ht2 stubs
wfname = PUFDIR + 'weights_rwt1_ipopt.csv'
weights = pd.read_csv(wfname)

# get national pufsums with these weights, for ht2 stubs
# these are the amounts we will share across states
pufsums_ht2 = rwp.get_wtdsums(pufsub, ptarget_names, weights, stubvar='ht2_stub')
pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')

# collapse ht2 shares to the states we want
compstates = ('NY', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX', 'VT')
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


# %% take a first cut at state weights
wfname1 = PUFDIR + 'weights_rwt1_ipopt.csv'
weights1 = pd.read_csv(wfname1)

# for testing
temp = pufsub.query('ht2_stub in [1, 2]').copy()
grouped = temp.groupby('ht2_stub')

# for real
grouped = pufsub.groupby('ht2_stub')

# uo = {'Q': Q_init, 'drops': drops, 'independent': True, 'max_iter': 10}
a = timer()
geo_weights = grouped.apply(gwp.get_geo_weights, weights1, targvars2, ht2wide, dropsdf_wide, independent=False)
b = timer()
b - a


# geo_weights[list(compstates) + ['other']].sum(axis=1)


# %% get new national weights by getting weights for each state (for each record) and summing them
wfname = PUFDIR + 'weights_rwt1_ipopt.csv'
weights = pd.read_csv(wfname)

grouped = pufsub.groupby('ht2_stub')

a = timer()
nat_geo_weights = grouped.apply(gwp.get_geo_weights, weights, targvars, ht2wide, dropsdf_wide, independent=True)
b = timer()
b - a

# note that the method here depends on coding within the function
wfname = PUFDIR + 'weights_geo_rwt.csv'
nat_geo_weights[['pid', 'geoweight_sum']].to_csv(wfname, index=None)

nat_geo_weights.to_csv(PUFDIR + 'weights_geo_independent.csv', index=None)

g = nat_geo_weights.geoweight_sum / nat_geo_weights.weight
np.quantile(g, qtiles)

nat_geo_weights.sum()


# %% create report on results with the geo revised national weights

# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interest
wfname1 = PUFDIR + 'weights_rwt1_ipopt.csv'
weights1 = pd.read_csv(wfname1)

# method = 'ipopt'  # ipopt or lsq
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
wfname = PUFDIR + 'weights_geo_rwt.csv'
comp_weights = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_geo_reweighted_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf ipopt method then georeweighted lsq, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_rwt=comp_weights,  # new_weights[['pid', 'reweight']],
                 weights_init=weights1,
                 targets=ptargets, outfile=rfname, title=rtitle)


# %% reweight the geo revised national weights
# in theory, these will be the weights we use to create the national file
# that will be shared out to states

wfname = PUFDIR + 'weights_geo_rwt.csv'
geo_weights = pd.read_csv(wfname).rename(columns={'geo_rwt': 'weight'})

method = 'ipopt'  # ipopt or lsq
drops = drops_ipopt

a = timer()
new_weights = rwp.puf_reweight(pufsub, geo_weights, ptargets, method=method, drops=drops)
b = timer()
b - a

wtname = 'georwt1_' +  method
wfname = PUFDIR + 'weights_georwt1_' + method + '.csv'
new_weights[['pid', 'reweight']].rename(columns={'reweight': wtname}).to_csv(wfname, index=None)


# %% create report on results with the revised georevised national weights
# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interest
wfname1 = PUFDIR + 'weights_geo_rwt.csv'
weights1 = pd.read_csv(wfname1)

# method = 'ipopt'  # ipopt or lsq
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
comp_weights = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_geo_reweighted_reweighted_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf ipopt method, georeweighted lsq, reweighted ipopt, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_rwt=comp_weights,  # new_weights[['pid', 'reweight']],
                 weights_init=weights1,
                 targets=ptargets, outfile=rfname, title=rtitle)


# %% construct new targets geoweight: get revised national weights based on independent state weights
# get weights to use as starting point for ht2 stubs
wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
weights = pd.read_csv(wfname)

# get national pufsums with these weights, for ht2 stubs
# these are the amounts we will share across states
pufsums_ht2 = rwp.get_wtdsums(pufsub, ptarget_names, weights, stubvar='ht2_stub')
pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')

# collapse ht2 shares to the states we want
compstates = ('NY', 'CA', 'CT', 'FL', 'MA', 'PA', 'NJ', 'TX', 'VT')
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
            # deductions
            'c17000','c17000_nnz',
            'c18300', 'c18300_nnz']
['good' for var in targvars if var in ht2_possible]

targvars2 = ['nret_all', 'mars1', 'c00100', 'e00200']


# %% run the final loop
# wfname1 = PUFDIR + 'weights_rwt1_ipopt.csv'
# weights1 = pd.read_csv(wfname1)

wfname = PUFDIR + 'weights_georwt1_ipopt.csv'
final_national_weights = pd.read_csv(wfname)

grouped = pufsub.groupby('ht2_stub')

a = timer()
geo_weights = grouped.apply(gwp.get_geo_weights, final_national_weights, targvars, ht2wide, dropsdf_wide, independent=False)
b = timer()
b - a

# geo_weights[list(compstates) + ['other']].sum(axis=1)
# note that the method here is lsq
#wfname = PUFDIR + 'weights_geo_rwt.csv'
# nat_geo_weights[['pid', 'geo_rwt']].to_csv(wfname, index=None)



# %% create report on results with the state weights



# %% create file with multiple national weights
# basenames of weight csv files
weight_list = ['weights_default', 'weights_regrown', 'weights_rwt1_lsq',
               'weights_rwt1_ipopt', 'weights_geo_rwt',
               'weights_georwt1_ipopt']
weight_df = rwp.merge_weights(weight_list, PUFDIR)  # they all must be in the same directory

weight_df.to_csv(PUFDIR + 'all_weights.csv', index=None)
weight_df.sum()


