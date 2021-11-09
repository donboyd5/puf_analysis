
# taxcalc's expected location for puf:
# '/home/donboyd/Documents/python_projects/Tax-Calculator/taxcalc/puf.csv'

# recs = tc.Records(data=puf,
#                 start_year=2011,
#                 gfactors=gfactors_object,
#                 weights=weights,
#                 adjust_ratios=adjust_ratios)

# Windows vs. Linux notes:
# note that I changed machine to windows in puf_constants.py


# %% imports
import numpy as np
import pandas as pd
import sys

import puf_utilities as pu

# TC_PATH = '/home/donboyd/Documents/python_projects/Tax-Calculator'
TC_PATH = 'C:/Users/donbo/Documents/python_projects/Tax-Calculator'
# TC_PATH = Path.home() / 'Documents/python_projects/Tax-Calculator'
# TC_DIR.exists()  # if not sure, check whether directory exists
# print("sys path before: ", sys.path)
if TC_PATH not in sys.path:
    sys.path.insert(0, str(TC_PATH))

import taxcalc as tc

# weights: string or Pandas DataFrame or None
#         string describes CSV file in which weights reside;
#         DataFrame already contains weights;
#         None creates empty sample-weights DataFrame;
#         default value is filename of the PUF weights.
#         NOTE: when using custom weights, set this argument to a DataFrame.
#         NOTE: assumes weights are integers that are 100 times the real weights.

# PATH=%PATH%;C:\Users\donbo\anaconda3;C:\Users\donbo\anaconda3\Scripts
# conda init cmd.exe
# %% locations
# inputs
#.. official puf weights
PUFWEIGHTS_DIR = r'E:\data\puf_files\puf_csv_related_files\PSL\2021-07-20/'

# my data, my state weights, and my reform files
INDATA_DIR = r'E:\puf_analysis_inputs\data/'
STATEWEIGHTS_DIR = r'E:\puf_analysis_inputs\weights/'
REFORM_DIR = 'C:/Users/donbo/Documents/python_projects/puf_analysis/reforms/'

# outputs
OUTDIR = r'E:\pufanalysis_outputs\salt_tc3.2.1/'


# %% constants
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% create puf_weights_alt.csv that has my weights for 2021 - simplest
# goal: create alt puf_weights file that
#   for 2017, uses my 2017 weights for filers, and puf_weights.csv weights for nonfilers
#   for 2021
#     filers - use my 2017 weights grown to 2021 at the same rate as total filers per puf_weights.csv
#     nonfilers - use 2021 weights from puf_weights.csv

# we will use this alt weights file to grow the puf to 2021, and never look at any other years


pufweights = pd.read_csv(PUFWEIGHTS_DIR + 'puf_weights.csv')
# create a RECID column
pufweights['RECID'] = np.arange(pufweights.shape[0]) + 1
pufweights.head()

# bring in my 2017 weights as REWT2017
weights2017 = pd.read_csv(STATEWEIGHTS_DIR + 'allweights2017_geo_restricted.csv') # 227766, filers only
weights2017['RECID'] = weights2017.pid + 1

weights2021 = pd.merge(pufweights.loc[:, ['RECID', 'WT2017', 'WT2021']], \
    weights2017.loc[:, ['RECID', 'weight']].rename(columns={'weight': 'REWT2017'}), \
        how='left', on='RECID')

# convert puf weights to decimal weights
weights2021.loc[:, ['WT2017', 'WT2021']] = weights2021.loc[:, ['WT2017', 'WT2021']] / 100.

# create filer2017 indicator for missing/nonmissing
weights2021['filer2017'] = weights2021.REWT2017.notna()
weights2021.filer2017.value_counts()  # good

# fill in REWT2017 with WT2017 where REWT2017 is missing
weights2021.loc[weights2021.filer2017==False, 'REWT2017'] = \
    weights2021.loc[weights2021.filer2017==False, 'WT2017']

# calculate rough 2021 weights as REWT2021
# apply a single ratio to all 2017 filer weights
pwratio = weights2021.loc[weights2021.filer2017==True, 'WT2021'].sum() \
    / weights2021.loc[weights2021.filer2017==True, 'WT2017'].sum()

# multiply 2017 filer reweights by the fixed ratio
weights2021.loc[weights2021.filer2017==True, 'REWT2021'] = \
    weights2021.loc[weights2021.filer2017==True, 'REWT2017'] * pwratio
weights2021.loc[weights2021.filer2017==False, 'REWT2021'] = \
    weights2021.loc[weights2021.filer2017==False, 'WT2021']

weights2021 = weights2021.loc[:, ['RECID', 'filer2017', 'WT2017', 'WT2021', 'REWT2017', 'REWT2021']]

wsums = weights2021.drop(labels='RECID', axis=1).sum(axis=0)
wsums



# %% get my 2017 version of puf, advance to 2021
# columns=['pid', 's006']  .rename(columns={'s006': 'weight'}
puf_base = pd.read_parquet(INDATA_DIR + 'puf2017.parquet', engine='pyarrow')  # 252868
puf_base.filer.value_counts()  # 227766 filers, 25102 nonfilers
puf_base.loc[puf_base.filer, ['pid', 'filer', 's006']]  # take a look at s006

# %% advance to 2021 using baseline law

# # put reweighted national weight on file
# puf = pd.merge(puf_base, weights.loc[:, ['pid', 'weight']], how='left', on='pid')
# # look at some values
# puf.loc[:, ['pid', 'filer', 's006', 'weight']]
# # save renamed s006 as s006_initial, replace s006 with weight where available
# puf['s006_initial'] = puf['s006']
# puf.loc[puf['weight'].notna(), 's006'] = puf['weight']
# puf.loc[:, ['RECID', 'pid', 'filer', 's006', 'weight', 's006_initial']]  # pid=RECIC - 1

# do one of the following
recs = tc.Records(data=puf_base, start_year=2017, adjust_ratios=None)
# recs = tc.Records()

pol = tc.Policy()
calc1 = tc.Calculator(policy=pol, records=recs)

# %% baseline
CYR = 2021
calc1.advance_to_year(CYR)
calc1.calc_all()
itax_rev1 = calc1.weighted_total('iitax')
itax_rev1 / 1e9
# we get different results depending on the commit
# 25 July 2021 commit 297931b7 Update test benchmark gives
# 1188.5434200603966 no adjust_ratios, 1189.4315467136157 with adjust ratios

# 11/8/2021 master gives 1197.7420925540068


# %% reform -- eliminate SALT cap
reform_filename = REFORM_DIR + 'reform_salt.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc2 = tc.Calculator(policy=pol, records=recs)
calc2.advance_to_year(CYR)
calc2.calc_all()
itax_rev2 = calc2.weighted_total('iitax')
itax_rev2 / 1e9
# 25 July 2021 commit 297931b7 Update test benchmark gives
# 1108.399314575875

# 11/8/2021 master gives:
#   1117.597987069486; 80.14410548452088


# %% reform -- set cap to $72,500
reform_filename = REFORM_DIR + 'reform_salt_72500.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc3 = tc.Calculator(policy=pol, records=recs)
calc3.advance_to_year(CYR)
calc3.calc_all()
itax_rev3 = calc3.weighted_total('iitax')
itax_rev3 / 1e9
# 25 July 2021 commit 297931b7 Update test benchmark gives
# 1131.67616383193

# 11/8/2021 master gives 1140.8748363255409; 56.867256228465976


# %% reform -- set cap to $80,000
reform_filename = REFORM_DIR + 'reform_salt_80000.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc4 = tc.Calculator(policy=pol, records=recs)
calc4.advance_to_year(CYR)
calc4.calc_all()
itax_rev4 = calc4.weighted_total('iitax')
itax_rev4 / 1e9
# 25 July 2021 commit 297931b7 Update test benchmark gives
# 1130.1879806428294

# 11/8/2021 master gives 1139.3866531364406


# %% comparison
# print('{}_CLP_itax_rev($B) = {:.3f}'.format(CYR, itax_rev1 * 1e-9))
# print('{}_REF_itax_rev($B) = {:.3f}'.format(CYR, itax_rev2 * 1e-9))
print('{}_CLP_itax_rev($B) = {:.6f}'.format(CYR, itax_rev1 * 1e-9))
print('{}_REF_itax_rev($B) = {:.6f}'.format(CYR, itax_rev2 * 1e-9))
# Matt's results (summer 2021 I think):
# 2021_CLP_itax_rev($B) = 1189.432
# 2021_REF_itax_rev($B) = 1109.253
# I get the same thing when I use all defaults

# what I get with adjust_ratios = None
# 2021_CLP_itax_rev($B) = 1188.543420
# 2021_REF_itax_rev($B) = 1108.399315

print('{}_DIFF_itax_rev($B) = {:.3f}'.format(CYR, (itax_rev2 - itax_rev1) * 1e-9))
# Matt's results
# 2021_DIFF_itax_rev($B) = -80.178

# what I get with adjust_ratios = None
# 2021_DIFF_itax_rev($B) = -80.144


# %% save all files
weights2021.to_csv(OUTDIR + 'weights2021.csv', index=False)

basedf = calc1.dataframe(variable_list=[], all_vars=True)
basedf.loc[:, ['RECID', 's006']] # this has the initial 2021 weights!

repeal_df = calc2.dataframe(variable_list=[], all_vars=True)
repeal_df.loc[:, ['RECID', 's006']] # this has the initial 2021 weights!
# what is the weighted value of the SALT deduction for millionaires??
((repeal_df.c00100 > 1e6) * repeal_df.s006 * repeal_df.c18300).sum() / 1e9
(repeal_df.s006 * repeal_df.c18300).sum() / 1e9

cap72500_df = calc3.dataframe(variable_list=[], all_vars=True)
cap80000_df = calc4.dataframe(variable_list=[], all_vars=True)

basedf.to_parquet(OUTDIR + 'base2021_2021.parquet', engine='pyarrow')
repeal_df.to_parquet(OUTDIR + 'repeal_2021.parquet', engine='pyarrow')
cap72500_df.to_parquet(OUTDIR + 'cap72500_2021.parquet', engine='pyarrow')
cap80000_df.to_parquet(OUTDIR + 'cap80000_2021.parquet', engine='pyarrow')
