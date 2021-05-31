

from collections import namedtuple
import numpy as np
import pandas as pd

import functions_advance_puf as adv
import puf_constants as pc
import taxcalc as tc
import puf_utilities as pu


def advance_and_save_puf(year, pufpath, growpath, wtpath, ratiopath, outdir):
    savepath=outdir + 'puf' + str(year) + '.parquet'

    print('getting puf.csv file...')
    puf = pd.read_csv(pufpath)

    print('creating records object...')
    gfactors_object = tc.GrowFactors(growpath)

    recs = tc.Records(data=puf,
                  start_year=2011,
                  gfactors=gfactors_object,
                  weights=wtpath,
                  adjust_ratios=ratiopath)
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
    print('All done!')


def get_potential_national_targets(targets_fname):
    targets_possible = pd.read_csv(targets_fname)

    possible_wide = targets_possible.loc[:, ['common_stub', 'pufvar', 'irs']] \
        .pivot(index='common_stub', columns='pufvar', values='irs') \
        .reset_index()
    possible_wide.columns.name = None

    ptarget_names = possible_wide.columns.tolist()
    ptarget_names.remove('common_stub')

    Result = namedtuple('Result', 'ptargets, ptarget_names')

    res = Result(
        ptargets=possible_wide,
        ptarget_names=ptarget_names)

    return res


def prep_puf(pufpath, targets):
    # puf = puf.copy()
    puf = pd.read_parquet(pufpath, engine='pyarrow')

    target_names = targets.columns.tolist()
    target_names.remove('common_stub')

    # get unique list of variables and check if they are all in the puf
    vars = [s.replace('pos', '').replace('neg', '').replace('_nnz', '') for s in target_names]
    vars = ulist(vars)

    # create lists of variables to be created, without suffixes
    pos = [s.replace('pos', '') for s in target_names if ('pos' in s) and ('nnz' not in s)]
    neg = [s.replace('neg', '') for s in target_names if ('neg' in s) and ('nnz' not in s)]
    nnz = [s.replace('_nnz', '') for s in target_names if 'nnz' in s]

    puf['common_stub'] = pd.cut(
        puf['c00100'],
        pc.COMMON_STUBS,
        labels=range(1, 19),
        right=False)
    # avoid categorical variable, it causes problems!
    puf['common_stub'] = puf.common_stub.astype('int64')

    puf['ht2_stub'] = pd.cut(
        puf['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)
    # avoid categorical variable, it causes problems!
    puf['ht2_stub'] = puf.ht2_stub.astype('int64')

    puf['nret_all'] = 1

    # marital status indicators
    puf['mars1'] = puf.MARS.eq(1)
    puf['mars2'] = puf.MARS.eq(2)
    puf['mars3'] = puf.MARS.eq(3)
    puf['mars4'] = puf.MARS.eq(4)
    puf['mars5'] = puf.MARS.eq(5)

    # create any additional needed puf vars
    puf['taxac_irs'] = np.maximum(0, puf.c09200 - puf.niit - puf.refund)

    for var in pos:
        puf[var + 'pos'] = puf[var] * puf[var].gt(0)

    for var in neg:
        puf[var + 'neg'] = puf[var] * puf[var].lt(0)

    for var in nnz:
        puf[var + '_nnz'] = puf[var].ne(0) * 1

    # safely drop columns we don't want to keep
    idvars = ['pid', 'filer', 'common_stub', 'ht2_stub']
    numvars = ['nret_all', 'mars1', 'mars2', 'mars3', 'mars4']
    keep_vars = idvars + numvars + target_names
    keep_vars = ulist(keep_vars)  # keeps unique names in case there is overlap with idvars

    return puf.loc[puf['filer'], keep_vars]


def save_pufweights(wtpath, outdir, years):
    # read the weights file, put a person id (pid) on it, divide by 100, and save individual years
    # weight files will have pid, weight, shortname as columns
    df = pd.read_csv(wtpath) # 252868 now 248591 records
    df = df.divide(100.0)
    df['pid'] = np.arange(len(df))

    for year in years:
        print(year)
        wname = 'WT' + str(year)
        weights = df.loc[:, ['pid', wname]].rename(columns={wname: 'weight'})
        shortname = 'weights' + str(year) + '_default'
        weights['shortname'] = shortname
        weights.to_csv(outdir + shortname + '.csv', index=None)

    print('Done creating weight files.')


def get_pufweights(wtpath, year):
    # read the weights file, put a person id (pid) on it, divide by 100, and save individual years
    # weight files will have pid, weight, shortname as columns
    df = pd.read_csv(wtpath) # 252868 now 248591 records
    df = df.divide(100.0)
    df['pid'] = np.arange(len(df))

    wname = 'WT' + str(year)
    weights = df.loc[:, ['pid', wname]].rename(columns={wname: 'weight'})
    shortname = 'weights' + str(year) + '_default'
    weights['shortname'] = shortname
    return weights


def ulist(thelist):
    # return unique list without changing order of original list
    ulist = []
    for x in thelist:
        if x not in ulist:
            ulist.append(x)
    return ulist