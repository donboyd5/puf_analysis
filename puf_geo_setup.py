# -*- coding: utf-8 -*-
"""
Create and save adjust puf and ht2 files that can be used to create
stub subsets for geo weighting.

@author: donbo
"""

# Notes about eclipse


# %% imports
import os
import sys
import requests
import pandas as pd
import numpy as np

import puf.puf_constants as pc


# %% set working directory if not already set
# os.getcwd()
# os.chdir('C:/programs_python/weighting')
# os.getcwd()


# %% constants and file locations

DOWNDIR = 'C:/programs_python/weighting/puf/downloads/'
DATADIR = 'C:/programs_python/weighting/puf/data/'
HDFDIR = 'C:/programs_python/weighting/puf/ignore/'
IGNOREDIR = 'C:/programs_python/weighting/puf/ignore/'

HT2 = DOWNDIR + '18in55cmagi.csv'
PUF_RWTD = HDFDIR + 'puf2018_reweighted.h5'


# %% program functions

def download_HT2(files, webdir, downdir):
    files = [files]
    for f in files:
        print(f)
        url = webdir + f
        path = downdir + f
        r = requests.get(url)
        print(r.status_code)

    with open(path, "wb") as file:
        file.write(r.content)


def get_geo_data():
    IGNOREDIR = 'C:/programs_python/weighting/puf/ignore/'

    ht2sub= pd.read_csv(IGNOREDIR + 'ht2sub.csv')
    ht2sums= pd.read_csv(IGNOREDIR + 'ht2sums.csv')

    pufsub = pd.read_csv(IGNOREDIR + 'pufsub.csv')
    pufsums= pd.read_csv(IGNOREDIR + 'pufsums.csv')
    return ht2sub, ht2sums, pufsub, pufsums


def prep_stub(stub, targvars, targstates, ht2sub, ht2sums, pufsub, pufsums):
    # get wh, xmat, and targets

    pufstub = pufsub.query('HT2_STUB == @stub')[['pid', 'HT2_STUB', 'wtnew'] + targvars]
    ptot = pufsums.query('HT2_STUB ==@stub')[targvars]

    ht2stub = ht2sub.query('HT2_STUB == @stub & STATE != "US"')[['STATE', 'HT2_STUB'] + targvars]
    htot = ht2sums.query('HT2_STUB ==@stub')[targvars]
    # ptot / htot

    # collapse ht2stub to target states and XX which will be all other
    mask = np.logical_not(ht2stub['STATE'].isin(targstates))
    ht2stub.loc[mask, 'STATE'] = 'XX'
    ht2stub = ht2stub.groupby(['STATE', 'HT2_STUB']).sum()

    # prepare the return values
    wh = pufstub.wtnew.to_numpy()
    xmat = np.asarray(pufstub[targvars], dtype=float)
    targets = np.asarray(ht2stub, dtype=float)
    return wh, xmat, targets


# %% ONETIME download Historical Table 2

# download_HT2(pc.HT2_2018, pc.WEBDIR, DOWNDIR)


# %% get HT2 and reweighted national puf
ht2 = pd.read_csv(DOWNDIR + pc.HT2_2018, thousands=',')
pufrw = pd.read_hdf(PUF_RWTD)  # 1 sec
# pufrw.columns.sort_values()


# %% prepare puf subset and weighted sums
# create a subset with ht2stub variable
pufsub = pufrw.copy()
pufsub['HT2_STUB'] = pd.cut(
    pufsub['c00100'],
    pc.HT2_AGI_STUBS,
    labels=list(range(1, len(pc.HT2_AGI_STUBS))),
    right=False)
pufsub.columns.sort_values().tolist()  # show all column names

# pufsub[['pid', 'c00100', 'HT2_STUB', 'IRS_STUB']].sort_values(by='c00100')

# create list of target vars
alltargvars = ['nret_all', 'nret_mars1', 'nret_mars2',
            'c00100', 'e00200', 'e00300', 'e00600',
            'c01000', 'e02400', 'c04800', 'irapentot']
alltargvars

pufsums = pufsub.groupby('HT2_STUB').apply(wsum,
                                            sumvars=alltargvars,
                                            wtvar='wtnew')
pufsums = pufsums.append(pufsums.sum().rename(0)).sort_values('HT2_STUB')
pufsums['HT2_STUB'] = pufsums.index
pufsums


# %% prepare compatible ht2 subset
# we'll rename HT2 columns to be like puf

# prepare a list of column names we want from ht2
ht2_all_names = list(pc.HT2PUF_XWALK.keys())  # superset
prefixes = ('N0')  # must be tuple, not a list
ht2_use_names = [x for x in ht2_all_names if not x.startswith(prefixes)]
drops = ['N17000', 'A17000', 'A04470', 'A05800', 'A09600', 'A00700']
ht2_use_names = [x for x in ht2_use_names if x not in drops]
ht2_use_names.append('STATE')
ht2_use_names.append('AGI_STUB')
ht2_use_names

# get those columns, rename as needed, and create new columns
ht2_sub = ht2.loc[:, ht2_use_names].copy()
ht2_sub = ht2_sub.rename(columns=pc.HT2PUF_XWALK)
ht2_sub = ht2_sub.rename(columns={'AGI_STUB': 'HT2_STUB'})
# multiply dollar values by 1000
dollcols = ['c00100', 'e00200', 'e00300',
            'e00600', 'c01000', 'e02400', 'c04800', 'irapentot']
ht2_sub[dollcols] = ht2_sub[dollcols] * 1000
ht2_sub


# %% compare pufsums to HT2 for US
ht2sums = ht2_sub.query('STATE=="US"').drop(columns=['STATE'])

# pd.options.display.max_columns = 99 # if need to see more; reset to 0 afterward
round(pufsums.drop(columns='HT2_STUB') / ht2sums.drop(columns='HT2_STUB') * 100 - 100)
# e02400 is way off, c04800 has some trouble, and irapentot is way off, so don't use them
# the rest look good


# %% create adjusted HT2 targets for all states, based on the ratios
# create adjustment ratios to apply to all ht2 values, based on national relationships
pufht2_ratios = pufsums / ht2sums
pufht2_ratios['HT2_STUB'] = pufht2_ratios.index
pufht2_ratios = pufht2_ratios.fillna(1)  # we won't use c04800
pufht2_ratios

# multiply each column of ht2_sub by its corresponding pufht2_ratios column
# is this the best way?
ht2_sub_adj = ht2_sub.copy()
ht2_sub_adjlong = pd.melt(ht2_sub_adj, id_vars=['HT2_STUB', 'STATE'])
ratios_long = pd.melt(pufht2_ratios, id_vars=['HT2_STUB'], value_name='ratio')
ht2_sub_adjlong =pd.merge(ht2_sub_adjlong, ratios_long, on=['HT2_STUB', 'variable'])
ht2_sub_adjlong['value'] = ht2_sub_adjlong['value'] * ht2_sub_adjlong['ratio']
ht2_sub_adjlong = ht2_sub_adjlong.drop(['ratio'], axis=1)
ht2_sub_adj = ht2_sub_adjlong.pivot(index=['HT2_STUB', 'STATE'], columns='variable', values='value')
# now we have an adjusted ht2 subset that has US totals equal to the puf totals

# check
pufsums / ht2_sub_adj.query('STATE=="US"')

ht2_sub_adj = ht2_sub_adj.reset_index() # get indexes as columns


# %% ONETIME save the adjusted files
pufsub.to_csv(IGNOREDIR + 'pufsub.csv', index=None)
pufsums.to_csv(IGNOREDIR + 'pufsums.csv', index=None)

ht2_sub_adj.to_csv(IGNOREDIR + 'ht2sub.csv', index=None)  # note name change
ht2sums.to_csv(IGNOREDIR + 'ht2sums.csv', index=None)  # note name change


# %% Now we are ready to get the geo data and prepare a stub
ht2sub, ht2sums, pufsub, pufsums = get_geo_data()

# %% choose a definition of targvars and targstates
targvars = pc.targvars_all
targstates = pc.STATES

badstates = ['AK', 'ND', 'SD', 'UT', 'WY']
targstates = [x for x in pc.STATES if x not in badstates]


# %% prepare a single stub for geoweighting
pufsub.columns
pufsub[['HT2_STUB', 'pid']].groupby(['HT2_STUB']).agg(['count'])

wh, xmat, targets = prep_stub(7, targvars, targstates,
                              ht2sub=ht2sub, ht2sums=ht2sums,
                              pufsub=pufsub, pufsums=pufsums)
wh.shape
xmat.shape
targets.shape


