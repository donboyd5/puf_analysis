# -*- coding: utf-8 -*-
"""
  Create two advanced files:
      one advanced in the normal way with all 3 stages
      one advanced with only stages 1 and 2, and with my alternative growfactors

  See Peter's code here:
      https://github.com/Peter-Metz/state_taxdata/blob/master/state_taxdata/prepdata.py

  List of official puf files:
      https://docs.google.com/document/d/1tdo81DKSQVee13jzyJ52afd9oR68IwLpYZiXped_AbQ/edit?usp=sharing
      Per Peter latest file is here (8/20/2020 as of 9/13/2020)
      https://www.dropbox.com/s/hyhalpiczay98gz/puf.csv?dl=0
      C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20

@author: donbo

"""

# %% imports
import taxcalc as tc
import pandas as pd
# pytables needed when using hdf
# conda install -c anaconda pytables
# I had to reboot machine and also exited spyder (not sure if needed) to
# install successfully
import numpy as np

from datetime import date

import puf_constants as pc

import puf_extrapolate_custom as xc
import puf_utilities as pu

# setup
# recs = tc.Records() # get the puf, not the cps version

# %% constants
# raw string allows Windows-style slashes
# note that it still cannot end with a single backslash - that must be forward slash

# directories
PUFDIR = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/'
DIR_FOR_OFFICIAL_PUF = PUFDIR + '2020-08-20/'
INDIR = PUFDIR + '2020-08-13_djb/'  # has the puf.csv that I created

DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
PUFOUTDIR = IGNOREDIR + 'puf_versions/'

# latest official puf per peter:
# PUF_NAME = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20\puf.csv'
# LATEST_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20\puf.csv'
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'

# latest version of the puf that I created with taxdata
BOYD_PUF = INDIR + 'puf.csv'
# r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

# growfactors
GF_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'growfactors.csv'
GF_ONES = DATADIR + 'growfactors_ones.csv'
# I developed custom growfactors that reflect IRS growth between 2011 and 2017
GF_CUSTOM = DATADIR + 'growfactors_custom.csv'

WEIGHTS_OFFICIAL = DIR_FOR_OFFICIAL_PUF + 'puf_weights.csv'


# %% date play
# today = date.today()
# today.strftime("%d/%m/%Y")  # dd/mm/YYYY
# today.strftime("%B %d, %Y")  # month name
# today.strftime("%m/%d/%y")  # mm/dd/yy
# today.strftime("%b-%d-%Y") # Month abbreviation, day and year
# today.strftime("%Y-%m-%d")  # suitable as file date identifier
# date_id = date.today().strftime("%Y-%m-%d")


# %% get puf
puf = pd.read_csv(LATEST_OFFICIAL_PUF)


# %% advance to 2017 in the normal (default) way -- all 3 stages
recs = tc.Records(data=puf, start_year=2011)  # start_year not needed for puf.csv
pol = tc.Policy()
calc = tc.Calculator(policy=pol, records=recs)
calc.advance_to_year(2017)
calc.calc_all()
puf2017_default = calc.dataframe(variable_list=[], all_vars=True)
puf2017_default['pid'] = np.arange(len(puf2017_default))

puf2017_default.to_parquet(PUFOUTDIR + 'puf2017_default' + '.parquet', engine='pyarrow')
check = pd.read_parquet(PUFOUTDIR + 'puf2017_default' + '.parquet', engine='pyarrow')
check.head(10)[['s006', 'c00100', 'e00300']]

pu.uvals(check.columns)



# %% advance to 2017 - CUSTOM growfactors
# extrapolate the underlying data with custom growfactors, BEFORE creating Records object
# then create record objects with a dummy set of growfactors equal to one so that
# tax-calculator won't extrapolate further (i.e., again)
gf_custom = pd.read_csv(GF_CUSTOM)
gfactor_ones = tc.GrowFactors(GF_ONES)
puf_extrap = xc.extrapolate_custom(puf, gf_custom, 2017)

recs_extrap = tc.Records(data=puf_extrap,
                  start_year=2011,
                  gfactors=gfactor_ones,
                  weights=WEIGHTS_OFFICIAL,
                  adjust_ratios=None)  # don't use puf_ratios

pol = tc.Policy()
calc_extrap = tc.Calculator(policy=pol, records=recs_extrap)
calc_extrap.advance_to_year(2017)
calc_extrap.calc_all()
puf2017_regrown = calc_extrap.dataframe(variable_list=[], all_vars=True)
puf2017_regrown['pid'] = np.arange(len(puf2017_regrown))

puf2017_regrown.to_parquet(PUFOUTDIR + 'puf2017_regrown' + '.parquet', engine='pyarrow')
check_regrown = pd.read_parquet(PUFOUTDIR + 'puf2017_regrown' + '.parquet', engine='pyarrow')
check_regrown.head(10)[['s006', 'c00100', 'e00300']]
check.head(10)[['s006', 'c00100', 'e00300']]


# %% advance 2017 file to 2018: default growfactors, no weights or ratios, then calculate 2018 law
puf2017_regrown = pd.read_parquet(PUFOUTDIR + 'puf2017_regrown' + '.parquet', engine='pyarrow')

# Note: advance does NOT extrapolate weights. It just picks the weights from the growfactors file
# puf2017_regrown.loc[puf2017_regrown.pid==0, 's006'] = 100

recs = tc.Records(data=puf2017_regrown,
                  start_year=2017,
                  adjust_ratios=None)

pol = tc.Policy()
calc = tc.Calculator(policy=pol, records=recs)
calc.advance_to_year(2018)
calc.calc_all()
puf2018 = calc.dataframe(variable_list=[], all_vars=True)
puf2018['pid'] = np.arange(len(puf2018))

puf2018.to_parquet(PUFOUTDIR + 'puf2018' + '.parquet', engine='pyarrow')

puf2018 = pd.read_parquet(PUFOUTDIR + 'puf2018' + '.parquet', engine='pyarrow')

puf2017_regrown.head(10)[['pid', 's006', 'c00100', 'e00200', 'e00300']]
puf2018.head(10)[['pid', 's006', 'c00100', 'e00200', 'e00300']]

puf2017_regrown.tail(10)[['pid', 's006', 'c00100', 'e00200', 'e00300']]
puf2018.tail(10)[['pid', 's006', 'c00100', 'e00200', 'e00300']]

# check.tail(10)[['pid', 's006', 'c00100', 'e00200', 'e00300']]




