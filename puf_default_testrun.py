# -*- coding: utf-8 -*-

# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/REFORMS.md#how-to-specify-a-tax-reform-in-a-json-policy-reform-file

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
REFORMSDIR = r'C:\programs_python\puf_analysis\reforms/'

IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
TCOUTDIR = IGNOREDIR + 'taxcalc_output/'


# %% constants
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'  # August 20, 2020 puf.csv

# reform files and locations I took them from
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/2017_law.json
law_2017 = REFORMSDIR + '2017_law.json'

# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/TCJA.json
law_2018 = REFORMSDIR + 'TCJA.json'


# %% get reforms
params_2017 = tc.Calculator.read_json_param_objects(law_2017, None)
params_2018 = tc.Calculator.read_json_param_objects(law_2018, None)


# %% get data
puf = pd.read_csv(LATEST_OFFICIAL_PUF)
recs_puf = tc.Records(data=puf)


# %% follow this
clp = tc.Policy()
clp.implement_reform(clp.read_json_reform(law_2017))
calc_clp = tc.Calculator(records=recs_puf, policy=clp)
calc_clp.advance_to_year(2018)
calc_clp.calc_all()
calc_clp.weighted_total('iitax')

ref=tc.Policy()
ref.implement_reform(ref.read_json_reform(law_2018))
calc_ref = tc.Calculator(records=recs_puf, policy=ref)
calc_ref.advance_to_year(2018)
calc_ref.calc_all()
calc_ref.weighted_total('iitax')

(calc_clp.weighted_total('iitax') - calc_ref.weighted_total('iitax')) / 1e9


# %% save results
clpdf = calc_clp.dataframe(variable_list=[], all_vars=True)
clpdf.to_parquet(TCOUTDIR + 'pufcsv_2017law.parquet', engine='pyarrow')

refdf = calc_ref.dataframe(variable_list=[], all_vars=True)
refdf.to_parquet(TCOUTDIR + 'pufcsv_2018law.parquet', engine='pyarrow')


# %% examine results
result = calc_clp.difference_table(calc_ref, 'weighted_deciles', 'iitax')
result

result.loc[['ALL'], ['tax_cut', 'tax_inc', 'tot_change']]
#         tax_cut    tax_inc  tot_change
# ALL  114.581427  14.328379 -162.431775
result.loc[:, ['tax_cut', 'tax_inc', 'tot_change']]

result.columns
result.loc[['ALL'], ['tax_cut', 'tax_inc', 'tot_change']]
result[['tax_cut']].sum() # 244.083881
result[['tax_inc']].sum()  # 30.951877
result[['tot_change']].sum()  # -396.367271




# %% start with 2017 law
del(pol) # be 100% sure this is gone

pol = tc.Policy()

# estimate 2017 law on 2018 data
pol.implement_reform(params_2017['policy'])
calc2017 = tc.Calculator(policy=pol, records=recs_puf)
calc2017.advance_to_year(2018)
calc2017.calc_all()
itax2017 = calc2017.weighted_total('iitax')  # 1719791464209.197


# %% estimate tax using puf.csv

pol = tc.Policy()

# estimate TCJA on 2018 data
pol.implement_reform(params_2018['policy'])
calc2018 = tc.Calculator(policy=pol, records=recs_puf)
calc2018.advance_to_year(2018)
calc2018.calc_all()
itax2018 = calc2018.weighted_total('iitax')

# estimate 2017 law on 2018 data
pol.implement_reform(params_2017['policy'])
calc2017 = tc.Calculator(policy=pol, records=recs_puf)
calc2017.advance_to_year(2018)
calc2017.calc_all()
itax2017 = calc2017.weighted_total('iitax')

itax2018  # 1557359689038.7046
itax2017  # 1719791464209.197
diff = itax2018 - itax2017  # $162 billion cut -162431775170.49243
pdiff = diff / itax2017 * 100   # 9.4% cut


# %% how did returns with 2017 agi > $1 million do?
df2017 = calc2017.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])
df2018 = calc2018.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])

sub2017 = df2017.query('c00100 >= 1e6')  # get 2017 agi millionaires
sub2018 = df2018.query('RECID in @sub2017.RECID') # get the same records from dfTCJA

# get weighted sums of tax
taxm2017 = (sub2017.iitax * sub2017.s006).sum()  # $524 billion 523957799640.81213
taxm2018 = (sub2018.iitax * sub2018.s006).sum()  # $528 billion 527798114885.58154
taxm2018 / taxm2017 * 100 - 100  # + 0.7%   0.7329436163374226

# get weighted sums of agi
agi2017 = (sub2017.c00100 * sub2017.s006).sum()
agi2018 = (sub2018.c00100 * sub2018.s006).sum()
agi2018 / agi2017 * 100 - 100  # +1.8%  1.7550784754322422


sub2017.sort_values(by='RECID').head(10)
sub2018.sort_values(by='RECID').head(10)


# %% estimate tax law using puf.csv

# estimate TCJA on 2018 data
pol_TCJA = tc.Policy()
pol_TCJA.implement_reform(params_TCJA['policy'])
calc_TCJA = tc.Calculator(policy=pol_TCJA, records=recs_puf)
calc_TCJA.advance_to_year(2018)
calc_TCJA.calc_all()
itax_TCJA = calc_TCJA.weighted_total('iitax')


# estimate 2017 law on 2018 data
pol_2017 = tc.Policy()
pol_2017.implement_reform(params_2017['policy'])
calc_2017 = tc.Calculator(policy=pol_2017, records=recs_puf)
calc_2017.advance_to_year(2018)
calc_2017.calc_all()
itax_2017 = calc_2017.weighted_total('iitax')


# tax comparison
itax_TCJA  # 1557359689038.7046
itax_2017  # 1719791464209.197
diff = itax_TCJA - itax_2017  # -162431775170.49243
pdiff = diff / itax_2017 * 100   # -9.444852969146618

# how did returns with 2017 agi > $1 million do?
df2017 = calc_2017.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])
dfTCJA = calc_TCJA.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])
sub2017 = df2017.query('c00100 >= 1e6')

# get the same records from dfTCJA
subTCJA = dfTCJA.query('RECID in @sub2017.RECID')

# get weighted sums of tax
taxm2017 = (sub2017.iitax * sub2017.s006).sum()
taxmTCJA = (subTCJA.iitax * subTCJA.s006).sum()

taxm2017  # 523957799640.81213
taxmTCJA  # 527798114885.58154
taxmTCJA / taxm2017 * 100 - 100  # + 0.7%

agi2017 = (sub2017.c00100 * sub2017.s006).sum()
agiTCJA = (subTCJA.c00100 * subTCJA.s006).sum()
agiTCJA / agi2017 * 100 - 100  # +1.8%




# %% older stuff

# estimate 2018 current law, built in, on 2018 data
pol_current = tc.Policy()
calc_current = tc.Calculator(policy=pol_current, records=recs_puf)
calc_current.advance_to_year(2018)
calc_current.calc_all()
itax_2018 = calc_current.weighted_total('iitax')
itax_2018  # 1557359689038.7046
