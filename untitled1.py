
# %% imports

from importlib import reload

import sys
from pathlib import Path

import pandas as pd

# importing taxcalc -- source code version
# soon use with the following
# TC_PATH = '/home/donboyd/Documents/python_projects/Tax-Calculator'
# if 'taxcalc' in sys.modules:
#     del sys.modules["taxcalc"]
TC_PATH = Path.home() / 'Documents/python_projects/Tax-Calculator'
# TC_DIR.exists()  # if not sure, check whether directory exists
if str(TC_PATH) not in sys.path:
    sys.path.insert(0, str(TC_PATH))

import taxcalc as tc


# %% reimports
# reload(tc)  # reload will not work with taxcalc because it is a package with imports


# %% constants
PUFDIR = r'/media/don/data/puf_files/puf_csv_related_files/Boyd/2021-07-02/'
WEIGHTDIR = PUFDIR
PUF_USE = PUFDIR + 'puf.csv'
GF_USE = PUFDIR + 'growfactors.csv'
WEIGHTS_USE = WEIGHTDIR + 'puf_weights.csv'
RATIOS_USE = PUFDIR + 'puf_ratios.csv'


# %% get data
puf = pd.read_csv(PUF_USE)
gfactors_object = tc.GrowFactors(GF_USE)

# %% test
recs = tc.Records(data=puf,
                start_year=2011,
                gfactors=gfactors_object,
                weights=WEIGHTS_USE,
                adjust_ratios=RATIOS_USE)

# %% test 2
recs = taxcalc.Records(data=puf,
                start_year=2011,
                gfactors=gfactors_object,
                weights=WEIGHTS_USE,
                adjust_ratios=RATIOS_USE)



# %%
