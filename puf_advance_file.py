# coding: utf-8
"""
Created on Sun Sep 13 06:33:21 2020

  # #!/usr/bin/env python
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

# setup
# recs = tc.Records() # get the puf, not the cps version

# %% constants
# raw string allows Windows-style slashes
# note that it still cannot end with a single backslash
PUFDIR = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/'

INDIR = PUFDIR + '2020-08-13_djb/'  # puf.csv that I created
DATADIR = r'C:\programs_python\puf_analysis\data/'

# r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

# latest version of the puf that I created with taxdata
PUF_NAME = INDIR + 'puf.csv'
GF_NAME = INDIR + 'growfactors.csv'
WEIGHTS_NAME = INDIR + 'puf_weights.csv'

# latest official puf per peter:
# PUF_NAME = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20\puf.csv'

# agi stubs
# AGI groups to target separately
IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]

HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]


# %% create objects
gfactor = tc.GrowFactors(GF_NAME)
dir(gfactor)

puf = pd.read_csv(PUF_NAME)

recs = tc.Records(data=puf,
                  start_year=2011,
                  gfactors=gfactor,
                  weights=WEIGHTS_NAME,
                  adjust_ratios=None)  # don't use puf_ratios

# recs = tc.Records(data=mypuf,
#                   start_year=2011,
#                   gfactors=gfactor,
#                   weights=WEIGHTS_NAME)  # apply built-in puf_ratios.csv

# %% advance the file
pol = tc.Policy()
calc = tc.Calculator(policy=pol, records=recs)
CYR = 2018
calc.advance_to_year(CYR)
calc.calc_all()


# %% create and examine data frame
puf_2018 = calc.dataframe(variable_list=[], all_vars=True)
puf_2018['pid'] = np.arange(len(puf_2018))

puf_2018.head(10)


# %% save advanced file
BASE_NAME = 'puf_adjusted'

# hdf5 is lightning fast
OUT_HDF = DATADIR + BASE_NAME + '.h5'
# %time puf_2018.to_hdf(OUT_HDF, key='puf_2018', mode='w')
%time puf_2018.to_hdf(OUT_HDF, 'data')  # 1 sec

# csv is slow, only use if need to share files
# OUT_CSV = DATADIR + BASE_NAME + '.csv'
# %time puf_2018.to_csv(OUT_CSV, index=False)  # 1+ minutes
# chunksize gives minimal speedup
# %time puf_2017.to_csv(OUT_NAME, index=False, chunksize=1e6)


# read back in
# %time dfcsv = pd.read_csv(OUT_CSV)  # 8 secs
%time dfhdf = pd.read_hdf(OUT_HDF)  # 1 sec
dfcsv.tail()
dfhdf.tail()
puf_2018.tail()

del(dfcsv)
del(dfhdf)
