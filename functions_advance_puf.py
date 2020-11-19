
import numpy as np
import pandas as pd
import taxcalc as tc
import puf_extrapolate_custom as xc
import puf_utilities as pu


def advance_puf(puf, year, savepath):
    print('creating records object...')
    recs = tc.Records(data=puf, start_year=2011)  # start_year not needed for puf.csv
    pol = tc.Policy()
    calc = tc.Calculator(policy=pol, records=recs)
    print(f'advancing puf to {year}...')
    calc.advance_to_year(year)
    print(f'calculating policy for {year}...')
    calc.calc_all()
    pufdf = calc.dataframe(variable_list=[], all_vars=True)
    pufdf['pid'] = np.arange(len(pufdf))
    pufdf['filer'] = pu.filers(pufdf)

    print('saving the advanced puf...')
    pufdf.to_parquet(savepath, engine='pyarrow')
    # no return
    return None


def advance_puf_custom(puf, year, gfcustom, gfones, weights, savepath):
    # extrapolate the underlying data with custom growfactors, BEFORE creating Records object
    # then create record objects with a dummy set of growfactors equal to one so that
    # tax-calculator won't extrapolate further (i.e., again)

    gfactor_custom = pd.read_csv(gfcustom)
    gfactor_ones = tc.GrowFactors(gfones)
    print(f'extrapolating puf to {year} with custom growfactors...')
    puf_extrap = xc.extrapolate_custom(puf, gfactor_custom, year)

    print('creating records object and advancing with dummy growfactors...')
    recs_extrap = tc.Records(data=puf_extrap,
                  start_year=2011,
                  gfactors=gfactor_ones,
                  weights=weights,
                  adjust_ratios=None)  # don't use puf_ratios

    pol = tc.Policy()
    calc_extrap = tc.Calculator(policy=pol, records=recs_extrap)
    calc_extrap.advance_to_year(year)
    calc_extrap.calc_all()
    pufdf_custom = calc_extrap.dataframe(variable_list=[], all_vars=True)
    pufdf_custom['pid'] = np.arange(len(pufdf_custom))
    pufdf_custom['filer'] = pu.filers(pufdf_custom)
    print(f'saving the custom-grown puf to {savepath}')
    pufdf_custom.to_parquet(savepath, engine='pyarrow')
    return None
