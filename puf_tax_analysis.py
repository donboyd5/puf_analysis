# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 06:57:20 2020

@author: donbo
"""


# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np
from datetime import date

import puf_constants as pc
import puf_utilities as pu

from timeit import default_timer as timer
from importlib import reload


# %% constants
puf2018 = pd.read_parquet(PUFDIR + 'puf2018_weighted' + '.parquet', engine='pyarrow')
sweights_2018 = pd.read_csv(PUFDIR + 'allweights2018_geo2017_grown.csv')

# check the weights
puf2018.loc[puf2018.pid==11, ['pid', 's006']]
puf2018[['pid', 's006']].head(20)
sweights_2018.head(5)


