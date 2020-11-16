
# %% imports
import sys
import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import functions_advance_puf as adv
import functions_reweight_puf as rwp
import puf_constants as pc

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


# %% names of files to create
PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)


# %% get puf.csv
puf = pd.read_csv(LATEST_OFFICIAL_PUF)


# %% ONETIME: create and save default and regrown 2017 pufs
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


# %% define any variable-stub combinations to drop via a drops dataframe
badvars = ['c02400', 'c02400_nnz']
stub1_4_vars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']
qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values
qx1 = "(pufvar in @badvars)"
qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @stub1_4_vars)"
qx = qxnan + " or " + qx1 + " or " + qx2
qx
drops = pdiff_init.query(qx).copy()
drops.sort_values(by=['common_stub', 'pufvar'], inplace=True)
drops


# %% reweight the puf file
method = 'ipopt'  # ipopt or lsq
a = timer()
new_weights = rwp.puf_reweight(pufsub, init_weights, ptargets, method=method, drops=drops)
b = timer()
b - a

wtname = 'rwt1_' +  method
wfname = PUFDIR + 'weights_rwt1_' + method + '.csv'
new_weights[['pid', 'reweight']].rename(columns={'reweight': wtname}).to_csv(wfname, index=None)


# %% check pdiffs
pdiff_rwt = rwp.get_pctdiffs(pufsub, new_weights[['pid', 'reweight']], ptargets)
pdiff_rwt.shape
pdiff_rwt.head(20)
pdiff_rwt.query('abspdiff > 10')


# %% create report on results from the reweighting
# CAUTION: a weights df must always contain only 2 variables, the first will be assumed to be
# pid, the second will be the weight of interst

# method = 'ipopt'  # ipopt or lsq
date_id = date.today().strftime("%Y-%m-%d")

# get weights for the comparison report
wfname = PUFDIR + 'weights_reweight1_' + method + '.csv'
comp_weights = pd.read_csv(wfname)

rfname = RESULTDIR + 'compare_irs_pufregrown_reweighted_' + method + '_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf, ' + method + ' method, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub,
                 weights_rwt=comp_weights,  # new_weights[['pid', 'reweight']],
                 weights_init=init_weights,
                 targets=ptargets, outfile=rfname, title=rtitle)


# %% get revised national weights based on independent construction of state weights


# %% create report on results with the revised national weights


# %% reweight the revised national weights


# %% create report on results with the revised national weights


# %% construct final state weights


# %% create report on results with the state weights



# %% create file with multiple national weights
# basenames of weight csv files
weight_list = ['weights_default', 'weights_regrown', 'weights_reweight1_lsq', 'weights_reweight1_ipopt']
weight_df = rwp.merge_weights(weight_list, PUFDIR)  # they all must be in the same directory

weight_df.to_csv(PUFDIR + 'all_weights.csv', index=None)
weight_df.sum()



