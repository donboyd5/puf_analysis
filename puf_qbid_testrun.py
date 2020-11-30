# -*- coding: utf-8 -*-

# To make the following suggestion, I'm just following along here
# (https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/calcfunctions.py#L811)
# and trying to prevent any limit from binding.

# Add three columns to your data file, `PT_SSTB_income`,  `PT_binc_w2_wages`,
# and `PT_ubia_property`.

# These are all accepted input variables, documented at the very bottom of this
# file (https://pslmodels.github.io/Tax-Calculator/guide/input_vars.html), so
# Tax-Calculator should accept them if found in your input datafile.

# PT_SSTB_income should be 0 for all filers. (Actually, you don't really need to
# add this column since Tax-Calculator treats missing tax input columns as zeroes
# anyways, but let's do it to be explicit).

# `PT_binc_w2_wages` should be 9e99 for all filers.
# (This will be multiplied by PT_qbid_w2_wages_rt,
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/policy_current_law.json#L12199
# which is 0.5 in relevant years, and ensure the wage cap
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/calcfunctions.py#L817
# is very high)

# `PT_ubia_property` should be 9e99 for all filers. (This will be multiplied by
# PT_qbid_alt_property_rt,
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/policy_current_law.json#L12307
# which is 0.25 in relevant years, and ensure that both alt_cap and full_cap
# are very high.)

# For this to work, I think we need to stay out of this conditional branch
# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/calcfunctions.py#L824
# because `adj` is not limited to zero or positive numbers, and that seems like
# it could be a problem if full_cap is huge.  So to prevent going in there, we
# can change the parameter value for `PT_qbid_taxinc_gap` to 0 for all MARS
# under both baseline and reform.

# https://github.com/PSLmodels/Tax-Calculator/blob/master/taxcalc/reforms/REFORMS.md#how-to-specify-a-tax-reform-in-a-json-policy-reform-file

# %% imports
import taxcalc as tc
import pandas as pd
import numpy as np


# %%  locations
DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
REFORMSDIR = r'C:\programs_python\puf_analysis\reforms/'


# %% constants
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'  # August 20, 2020 puf.csv

law_xqbid = REFORMSDIR + 'noQBIDlimit.json'


# %% get data
puf = pd.read_csv(LATEST_OFFICIAL_PUF)
recs_puf = tc.Records(data=puf)

clp = tc.Policy()
# clp.implement_reform(ref.read_json_reform(law_xqbid))
calc_clp = tc.Calculator(records=recs_puf, policy=clp)
calc_clp.advance_to_year(2018)
calc_clp.calc_all()
calc_clp.weighted_total('iitax') / 1e9

ref=tc.Policy()
ref.implement_reform(ref.read_json_reform(law_xqbid))
calc_ref = tc.Calculator(records=recs_puf, policy=ref)
calc_ref.advance_to_year(2018)
calc_ref.calc_all()
calc_ref.weighted_total('iitax') / 1e9

calc_clp.weighted_total('iitax') / 1e9
calc_ref.weighted_total('iitax') / 1e9
(calc_clp.weighted_total('iitax') - calc_ref.weighted_total('iitax')) / 1e9


# %% create current law and reform
clp = tc.Policy()
calc_clp = tc.Calculator(records=recs_puf, policy=clp)
calc_clp.advance_to_year(2018)
calc_clp.calc_all()
calc_clp.weighted_total('iitax') / 1e9

ref=tc.Policy()
ref.implement_reform(ref.read_json_reform(law_xqbid))
calc_ref = tc.Calculator(records=recs_puf, policy=ref)
calc_ref.advance_to_year(2018)
calc_ref.calc_all()
calc_ref.weighted_total('iitax') / 1e9

calc_clp.weighted_total('iitax') / 1e9
calc_ref.weighted_total('iitax') / 1e9
(calc_clp.weighted_total('iitax') - calc_ref.weighted_total('iitax')) / 1e9
