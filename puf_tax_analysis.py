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


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFDIR = IGNOREDIR + 'puf_versions/'
RESULTDIR = r'C:\programs_python\puf_analysis\results/'


# %% constants


# %% get data
puf2018 = pd.read_parquet(PUFDIR + 'puf2018_weighted' + '.parquet', engine='pyarrow')
sweights_2018 = pd.read_csv(PUFDIR + 'allweights2018_geo2017_grown.csv')

# check the weights
puf2018.loc[puf2018.pid==11, ['pid', 's006']]
puf2018[['pid', 's006']].head(20)
sweights_2018.head(5)


# %% run 2017 law and 2018 law on the file
