
# taxcalc's expected location for puf:
# '/home/donboyd/Documents/python_projects/Tax-Calculator/taxcalc/puf.csv'

# recs = tc.Records(data=puf,
#                 start_year=2011,
#                 gfactors=gfactors_object,
#                 weights=weights,
#                 adjust_ratios=adjust_ratios)


# %% imports
import numpy as np
import pandas as pd

import puf_utilities as pu

TC_PATH = '/home/donboyd/Documents/python_projects/Tax-Calculator'
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


# %% locations
DIR = '/media/don/pufanalysis_output/'
DDIR = DIR + 'data/'
WDIR = DIR + 'weights/'

SCRATCHDIR = '/media/don/scratch/salt_v2/'  # changed from original salt analysis
REFORM_DIR = '/home/donboyd/Documents/python_projects/puf_analysis/reforms/'


# %% constants
qtiles = (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)


# %% create puf_weights_alt.csv that has my weights for 2021 - simplest
# goal: create alt puf_weights file that
#   for 2017, uses my 2017 weights for filers, and puf_weights.csv weights for nonfilers
#   for 2021
#     filers - use my 2017 weights grown to 2021 at the same rate as total filers per puf_weights.csv
#     nonfilers - use 2021 weights from puf_weights.csv

# we will use this alt weights file to grow the puf to 2021, and never look at any other years

PWDIR = '/media/don/data/puf_files/puf_csv_related_files/PSL/2021-07-20/'
pufweights = pd.read_csv(PWDIR + 'puf_weights.csv')
# create a RECID column
pufweights['RECID'] = np.arange(pufweights.shape[0]) + 1
pufweights.head()

# bring in my 2017 weights as REWT2017
weights2017 = pd.read_csv(WDIR + 'allweights2017_geo_restricted.csv') # 227766, filers only
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
puf_base = pd.read_parquet(DDIR + 'puf2017.parquet', engine='pyarrow')  # 252868
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
itax_rev1 / 1e9  # 1188.5434200603966 no adjust_ratios, 1189.4315467136157 with adjust ratios



# %% reform -- eliminate SALT cap
reform_filename = REFORM_DIR + 'reform_salt.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc2 = tc.Calculator(policy=pol, records=recs)
calc2.advance_to_year(CYR)
calc2.calc_all()
itax_rev2 = calc2.weighted_total('iitax')
itax_rev2 / 1e9  # 1108.399314575875


# %% reform -- set cap to $72,500
reform_filename = REFORM_DIR + 'reform_salt_72500.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc3 = tc.Calculator(policy=pol, records=recs)
calc3.advance_to_year(CYR)
calc3.calc_all()
itax_rev3 = calc3.weighted_total('iitax')
itax_rev3 / 1e9  # 1131.67616383193


# %% reform -- set cap to $80,000
reform_filename = REFORM_DIR + 'reform_salt_80000.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc4 = tc.Calculator(policy=pol, records=recs)
calc4.advance_to_year(CYR)
calc4.calc_all()
itax_rev4 = calc4.weighted_total('iitax')
itax_rev4 / 1e9  # 1130.1879806428294



# %% comparison
# print('{}_CLP_itax_rev($B) = {:.3f}'.format(CYR, itax_rev1 * 1e-9))
# print('{}_REF_itax_rev($B) = {:.3f}'.format(CYR, itax_rev2 * 1e-9))
print('{}_CLP_itax_rev($B) = {:.6f}'.format(CYR, itax_rev1 * 1e-9))
print('{}_REF_itax_rev($B) = {:.6f}'.format(CYR, itax_rev2 * 1e-9))
# Matt's results:
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
weights2021.to_csv(SCRATCHDIR + 'weights2021.csv', index=False)

basedf = calc1.dataframe(variable_list=[], all_vars=True)
basedf.loc[:, ['RECID', 's006']] # this has the initial 2021 weights!

repeal_df = calc2.dataframe(variable_list=[], all_vars=True)
repeal_df.loc[:, ['RECID', 's006']] # this has the initial 2021 weights!
# what is the weighted value of the SALT deduction for millionaires??
((repeal_df.c00100 > 1e6) * repeal_df.s006 * repeal_df.c18300).sum() / 1e9
(repeal_df.s006 * repeal_df.c18300).sum() / 1e9


cap72500_df = calc3.dataframe(variable_list=[], all_vars=True)
cap80000_df = calc4.dataframe(variable_list=[], all_vars=True)


basedf.to_parquet(SCRATCHDIR + 'base2021_2021.parquet', engine='pyarrow')
repeal_df.to_parquet(SCRATCHDIR + 'repeal_2021.parquet', engine='pyarrow')
cap72500_df.to_parquet(SCRATCHDIR + 'cap72500_2021.parquet', engine='pyarrow')
cap80000_df.to_parquet(SCRATCHDIR + 'cap80000_2021.parquet', engine='pyarrow')


# %% quick check: what happened to e18400, which should have grown by ATXPY between 2017 and 2021
temp = calc1.dataframe(variable_list=[], all_vars=True)

puf_base.loc[range(10), 'e18400']
temp.loc[range(10), 'e18400']

temp.loc[range(10), 'e18400'] / puf_base.loc[range(10), 'e18400']

puf_base.loc[range(10), 'e18400']

# %% checks
recs0 = tc.Records(adjust_ratios=None)
pol0 = tc.Policy()
calc0 = tc.Calculator(policy=pol0, records=recs0)
calc0.advance_to_year(2021)
calc0.calc_all()
itax_rev0 = calc0.weighted_total('iitax')

# how much does salt-cap removal cost increase from 2020 to 2021 with all defaults?
params2020 = tc.Calculator.read_json_param_objects(REFORM_DIR + 'reform_salt_2020.json', None)
params2021 = tc.Calculator.read_json_param_objects(REFORM_DIR + 'reform_salt.json', None)

# baseline data
recs0 = tc.Records()
pol0 = tc.Policy()
calc0 = tc.Calculator(policy=pol0, records=recs0)

# 2020 baseline all defaults
calc0.advance_to_year(2020)
calc0.calc_all()
itax_rev2020_base = calc0.weighted_total('iitax')

# 2021 baseline all defaults
calc0.advance_to_year(2021)
calc0.calc_all()
itax_rev2021_base = calc0.weighted_total('iitax')

# 2020 reform all defaults
pol0.implement_reform(params2020['policy'])
calcref = tc.Calculator(policy=pol0, records=recs0)
calcref.advance_to_year(2020)
calcref.calc_all()
itax_rev2020_reform = calcref.weighted_total('iitax')

# 2021 reform all defaults
pol0.implement_reform(params2021['policy'])
calcref = tc.Calculator(policy=pol0, records=recs0)
calcref.advance_to_year(2021)
calcref.calc_all()
itax_rev2021_reform = calcref.weighted_total('iitax')

# reform costs
cost2020 = itax_rev2020_reform - itax_rev2020_base
cost2021 = itax_rev2021_reform - itax_rev2021_base

# show results
print('Cost 2020($B) = {:.6f}'.format(cost2020 * 1e-9))
print('Cost 2021($B) = {:.6f}'.format(cost2021 * 1e-9))
print('$b change cost = {:.2f}'.format((cost2021 - cost2020) * 1e-9))
print('% change cost = {:.2f}'.format(cost2021 / cost2020 * 100 - 100))


# c18300 in 2011
recs0 = tc.Records()
pol0 = tc.Policy()
calc0 = tc.Calculator(policy=pol0, records=recs0)
calc0.calc_all()
salt2011 = calc0.weighted_total('c18300')
print('salt 2011 ($B) = {:.6f}'.format(salt2011 * 1e-9))
