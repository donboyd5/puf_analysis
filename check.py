# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:28:17 2020

@author: donbo

https://github.com/PSLmodels/Tax-Calculator/blob/62f7e21bac6dd582e029c101a6193734a8e296ca/docs/recipes/md_src/recipe01.md

"""

import pandas as pd
import taxcalc as tc


DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
LATEST_OFFICIAL_PUF = DIR_FOR_OFFICIAL_PUF + 'puf.csv'  # August 20, 2020 puf.csv

# read an "old" reform file
# ("old" means the reform file is defined relative to pre-TCJA policy)
# REFORMS_PATH = '../../taxcalc/reforms/'
REFORMS_PATH = r'C:\programs_python\puf_analysis\reforms/'

# specify reform dictionary for pre-TCJA policy
reform1 = tc.Policy.read_json_reform(REFORMS_PATH + '2017_law.json')

# specify reform dictionary for TCJA as passed by Congress in late 2017
reform2 = tc.Policy.read_json_reform(REFORMS_PATH + 'TCJA.json')

# specify Policy object for pre-TCJA policy
bpolicy = tc.Policy()
bpolicy.implement_reform(reform1, print_warnings=False, raise_errors=False)
assert not bpolicy.parameter_errors

# specify Policy object for TCJA reform relative to pre-TCJA policy
rpolicy = tc.Policy()
rpolicy.implement_reform(reform1, print_warnings=False, raise_errors=False)
assert not rpolicy.parameter_errors
rpolicy.implement_reform(reform2, print_warnings=False, raise_errors=False)
assert not rpolicy.parameter_errors


puf = pd.read_csv(LATEST_OFFICIAL_PUF)

# recs = tc.Records.cps_constructor()  # use cps

recs = tc.Records(data=puf)  # or use puf

# specify Calculator objects using bpolicy and rpolicy
calc1 = tc.Calculator(policy=bpolicy, records=recs)
calc2 = tc.Calculator(policy=rpolicy, records=recs)

CYR = 2018

# calculate for specified CYR
calc1.advance_to_year(CYR)
calc1.calc_all()
calc2.advance_to_year(CYR)
calc2.calc_all()

# compare aggregate individual income tax revenue in cyr
iitax_rev1 = calc1.weighted_total('iitax')  # bpolicy 2017 law
iitax_rev2 = calc2.weighted_total('iitax')  # rpolicy

iitax_rev2 / iitax_rev1 * 100 - 100


df2017 = calc1.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])
(df2017.iitax * df2017.s006).sum() - iitax_rev1
df2018 = calc2.dataframe(variable_list=['RECID', 's006', 'c00100', 'iitax'])
((df2017.iitax * df2017.s006).sum()  - (df2018.iitax * df2018.s006).sum()) / 1e9

sub2017 = df2017.query('c00100 >= 1e6')  # get 2017 agi millionaires
sub2018 = df2018.query('RECID in @sub2017.RECID') # get the same records from dfTCJA
mtax2017 = (sub2017.iitax * sub2017.s006).sum()
mtax2018 = (sub2018.iitax * sub2018.s006).sum()
(mtax2018 - mtax2017) / 1e9

# construct reform-vs-baseline difference table with results for income deciles standard_income_bins
diff_table = calc1.difference_table(calc2, 'weighted_deciles', 'iitax')
diff_table.loc[:, ['count', 'tax_cut', 'tax_inc', 'tot_change']]

diff_table2 = calc1.difference_table(calc2, 'standard_income_bins', 'iitax')
diff_table2.loc[['>$1000K'], ['count', 'tax_cut', 'tax_inc', 'tot_change']]
diff_table2.loc[:, ['count', 'tax_cut', 'tax_inc', 'tot_change']]

diff_table3 = calc1.difference_table(calc2, 'soi_agi_bins', 'iitax')
diff_table3.loc[:, ['count', 'tax_cut', 'tax_inc', 'tot_change']]
# SOI_AGI_BINS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
#                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]

assert isinstance(diff_table, pd.DataFrame)
diff_extract = pd.DataFrame()
dif_colnames = ['count', 'tax_cut', 'tax_inc',
                'tot_change', 'mean', 'pc_aftertaxinc']
ext_colnames = ['funits(#m)', 'taxfall(#m)', 'taxrise(#m)',
                'agg_diff($b)', 'mean_diff($)', 'aftertax_income_diff(%)']
for dname, ename in zip(dif_colnames, ext_colnames):
    diff_extract[ename] = diff_table[dname]

# print total revenue estimates for cyr
# (estimates in billons of dollars)
print('{}_REFORM1_iitax_rev($B)= {:.3f}'.format(CYR, iitax_rev1 * 1e-9))  # 2017
print('{}_REFORM2_iitax_rev($B)= {:.3f}'.format(CYR, iitax_rev2 * 1e-9))  # 2018
print('')

title = 'Extract of {} income-tax difference table by expanded-income decile'
print(title.format(CYR))
print('(taxfall is count of funits with cut in income tax in reform 2 vs 1)')
print('(taxrise is count of funits with rise in income tax in reform 2 vs 1)')
print(diff_extract.to_string())