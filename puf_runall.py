
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


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'


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


# %% ONETIME: create default and regrown 2017 pufs
adv.advance_puf(puf, 2017, PUF_DEFAULT)

adv.advance_puf_custom(puf, 2017,
                       gfcustom=GF_CUSTOM, gfones=GF_ONES,
                       weights=WEIGHTS_OFFICIAL,
                       savepath=PUF_REGROWN)


# %% define targets
ptargets = rwp.get_possible_targets(POSSIBLE_TARGETS)
ptargets.columns
ptargets

target_names = ptargets.columns.tolist()
target_names.remove('common_stub')


# winnow down the targets list if desired


# %% prepare a version of the puf for reweighting
# do the following:
#   add filer and stub variables
#   create mars1, mars2, ... marital status indicators
#   create any positive or negative variables needed
#   create any needed nnz indicators
#   keep only the needed variables pid, common_stub,


# %% reweight the 2017 regrown puf
pufrg = pd.read_parquet(PUF_REGROWN, engine='pyarrow')
check = rwp.prep_puf(pufrg, target_names)
check.columns

a = ['d', 'b']
b = ['b', 'c']
ab = a + b
list(set(ab))
np.unique(np.array(ab))


[x in ab if x not in x]
l = []
for x in ab:
    if x not in l:
        l.append(x)

l




check
check.columns
pufrg.columns
puf.columns
puf is pufrg
check is pufrg

# %% create report on results from the reweighting
