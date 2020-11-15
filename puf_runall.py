
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


# %% define targets
ptargets = rwp.get_possible_targets(POSSIBLE_TARGETS)
ptargets.columns
ptargets

# winnow down the targets list if desired


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


# %% add initial weight to the file
weights = pd.read_csv(PUFDIR + 'weights_regrown.csv').rename(columns={'s006_regrown': 'weight'})
init_weights = weights.copy()
pufsub = pd.merge(pufsub.drop(columns='weight', errors='ignore'), weights, on='pid', how='left')
pufsub.shape
pufsub.columns


# %% reweight the puf file
a = timer()
new_weights = rwp.puf_reweight(pufsub, ptargets, method='ipopt')
b = timer()
b - a

# use one or the other of the following
# new_weights[['pid', 'reweight']].rename(columns={'reweight': 'rwt1_lsq'}).to_csv(PUFDIR + 'weights_reweight1_lsq.csv', index=None)
new_weights[['pid', 'reweight']].rename(columns={'reweight': 'rwt1_ipopt'}).to_csv(PUFDIR + 'weights_reweight1_ipopt.csv', index=None)


# %% create report on results from the reweighting
date_id = date.today().strftime("%Y-%m-%d")

# comp_weights must have pid and weight
# comp_weights = pd.read_csv(PUFDIR + 'weights_reweight1_lsq.csv').rename(columns={'rwt1_lsq': 'weight'})
# fname = RESULTDIR + 'compare_irs_pufregrown_reweighted_lsq_' + date_id + '.txt'
# rtitle = 'Regrown reweighted puf, lsq method, compared to IRS values, run on ' + date_id
# rwp.comp_report(pufsub=pufsub, weights=comp_weights, weights_init=init_weights, targets=ptargets, outfile=fname, title=rtitle)

comp_weights = pd.read_csv(PUFDIR + 'weights_reweight1_ipopt.csv').rename(columns={'rwt1_ipopt': 'weight'})
fname = RESULTDIR + 'compare_irs_pufregrown_reweighted_ipopt_' + date_id + '.txt'
rtitle = 'Regrown reweighted puf, ipopt method, compared to IRS values, run on ' + date_id
rwp.comp_report(pufsub=pufsub, weights=comp_weights, weights_init=init_weights, targets=ptargets, outfile=fname, title=rtitle)



# %% create file with multiple national weights
# basenames of weight csv files
weight_list = ['weights_default', 'weights_regrown', 'weights_reweight1_lsq', 'weights_reweight1_ipopt']
weight_df = rwp.merge_weights(weight_list, PUFDIR)  # they all must be in the same directory

weight_df.to_csv(PUFDIR + 'all_weights.csv', index=None)
weight_df.sum()



# %% get revised national weights based on independent construction of state weights


# %% create report on results with the revised national weights


# %% reweight the revised national weights


# %% create report on results with the revised national weights


# %% construct final state weights



# %% create report on results with the state weights



