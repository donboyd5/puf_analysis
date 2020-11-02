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

from datetime import date

import puf_constants as pc

import puf_extrapolate_custom as xc

# setup
# recs = tc.Records() # get the puf, not the cps version

# %% constants
# raw string allows Windows-style slashes
# note that it still cannot end with a single backslash
PUFDIR = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/'

INDIR = PUFDIR + '2020-08-13_djb/'  # puf.csv that I created
DATADIR = r'C:\programs_python\puf_analysis\data/'
IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'

# r'C:\Users\donbo\Downloads\taxdata_stuff\puf_2017_djb.csv'

# latest version of the puf that I created with taxdata
PUF_NAME = INDIR + 'puf.csv'

GF_NAME = INDIR + 'growfactors.csv'
GF_ONES = DATADIR + 'growfactors_ones.csv'
GF_CUSTOM = DATADIR + 'growfactors_custom.csv'

WEIGHTS_NAME = INDIR + 'puf_weights.csv'

# latest official puf per peter:
# PUF_NAME = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011\2020-08-20\puf.csv'


# %% date play
# today = date.today()
# today.strftime("%d/%m/%Y")  # dd/mm/YYYY
# today.strftime("%B %d, %Y")  # month name
# today.strftime("%m/%d/%y")  # mm/dd/yy
# today.strftime("%b-%d-%Y") # Month abbreviation, day and year
# today.strftime("%Y-%m-%d")  # suitable as file date identifier
# date_id = date.today().strftime("%Y-%m-%d")


# %% get puf
puf = pd.read_csv(PUF_NAME)


# %% alternative - CUSTOM growfactors

gf_custom = pd.read_csv(GF_CUSTOM)
gfactor_ones = tc.GrowFactors(GF_ONES)

puf_extrap = xc.extrapolate_custom(puf, gf_custom, 2017)

recs = tc.Records(data=puf_extrap,
                  start_year=2011,
                  gfactors=gfactor_ones,
                  weights=WEIGHTS_NAME,
                  adjust_ratios=None)  # don't use puf_ratios


# %% alternative - use original growfactors
gfactor = tc.GrowFactors(GF_NAME)
dir(gfactor)

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
# what happens if we advance twice?
pol = tc.Policy()
calc = tc.Calculator(policy=pol, records=recs)
CYR = 2017
calc.advance_to_year(CYR)
calc.calc_all()


# %% create and examine data frame
puf_advanced = calc.dataframe(variable_list=[], all_vars=True)
puf_advanced['pid'] = np.arange(len(puf_advanced))

puf.head()['e00200']
puf_extrap.head()['e00200']
puf_advanced.head(10)['e00200']


# %% save advanced file
date_id = date.today().strftime("%Y-%m-%d")

BASE_NAME = 'puf' + str(CYR) + '_' + date_id

# hdf5 is lightning fast
OUT_HDF = IGNOREDIR + BASE_NAME + '.h5'
puf_advanced.to_hdf(OUT_HDF, 'data')  # 1 sec

# csv is slow, only use if need to share files
# OUT_CSV = IGNOREDIR + BASE_NAME + '.csv'
# puf_advanced.to_csv(OUT_CSV, index=False)  # 1+ minutes
# chunksize gives minimal speedup
# %time puf_2017.to_csv(OUT_NAME, index=False, chunksize=1e6)


# read back in
# %time dfcsv = pd.read_csv(OUT_CSV)  # 8 secs
dfhdf = pd.read_hdf(OUT_HDF)  # 1 sec
# dfcsv.tail()
# dfhdf.tail()
puf_advanced.tail()

del(dfcsv)
del(dfhdf)
