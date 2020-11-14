
# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import functions_advance_puf
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


# %% names of files to create
PUF_DEFAULT = PUFDIR + 'puf2017_default.parquet'
PUF_REGROWN = PUFDIR + 'puf2017_regrown.parquet'


# %% constants
qtiles = (0, .01, .1, .25, .5, .75, .9, .99, 1)
pc.IRS_AGI_STUBS

# %% get puf.csv
puf = pd.read_csv(LATEST_OFFICIAL_PUF)


# %% create default 2017 puf
advance_puf(puf, 2017, PUF_DEFAULT)
advance_puf_custom(puf, 2017, gfcustom=GF_CUSTOM, gfones=GF_ONES, weights=WEIGHTS_OFFICIAL,
                   savepath=PUF_REGROWN)


# %% create regrown 2017 puf with custom growfactors
