
# taxcalc's expected location for puf:
# '/home/donboyd/Documents/python_projects/Tax-Calculator/taxcalc/puf.csv'

# recs = tc.Records(data=puf,
#                 start_year=2011,
#                 gfactors=gfactors_object,
#                 weights=weights,
#                 adjust_ratios=adjust_ratios)


# %% imports
TC_PATH = '/home/donboyd/Documents/python_projects/Tax-Calculator'
# TC_PATH = Path.home() / 'Documents/python_projects/Tax-Calculator'
# TC_DIR.exists()  # if not sure, check whether directory exists
# print("sys path before: ", sys.path)
if TC_PATH not in sys.path:
    sys.path.insert(0, str(TC_PATH))

import taxcalc as tc

# %% records object
recs = tc.Records()

pol = tc.Policy()
calc1 = tc.Calculator(policy=pol, records=recs)

# %% baseline
CYR = 2021
calc1.advance_to_year(CYR)
calc1.calc_all()
itax_rev1 = calc1.weighted_total('iitax')

# %% reform
REFORM_DIR = '/home/donboyd/Documents/python_projects/puf_analysis/reforms/'
# reform_filename = 'reform_salt.json'
reform_filename = REFORM_DIR + 'reform_salt.json'

params = tc.Calculator.read_json_param_objects(reform_filename, None)
pol.implement_reform(params['policy'])
calc2 = tc.Calculator(policy=pol, records=recs)
calc2.advance_to_year(CYR)
calc2.calc_all()
itax_rev2 = calc2.weighted_total('iitax')


# %% comparison
print('{}_CLP_itax_rev($B) = {:.3f}'.format(CYR, itax_rev1 * 1e-9))
print('{}_REF_itax_rev($B) = {:.3f}'.format(CYR, itax_rev2 * 1e-9))
# Matt's results:
# 2021_CLP_itax_rev($B) = 1189.432
# 2021_REF_itax_rev($B) = 1109.253

# what I get, which obviously is wrong:
# 2021_CLP_itax_rev($B) = 4068.671
# 2021_REF_itax_rev($B) = 3879.942

print('{}_DIFF_itax_rev($B) = {:.3f}'.format(CYR, (itax_rev2 - itax_rev1) * 1e-9))
# Matt's results
# 2021_DIFF_itax_rev($B) = -80.178

# what I get

print('{}_DIFF_itax_rev($B) = {:.3f}'.format(CYR, (itax_rev2 - itax_rev1) * 1e-9))
# 2021_DIFF_itax_rev($B) = -188.729