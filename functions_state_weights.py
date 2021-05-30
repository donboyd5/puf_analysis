
import numpy as np
import pandas as pd

import functions_advance_puf as adv


def advance_and_save_puf(year, pufpath, growpath, wtpath, ratiopath, outdir):
    puf = pd.read_csv(pufpath)
    # pufvars = puf.columns.tolist()

    # just need to create the advanced puf files once
    adv.advance_puf2(puf, year=year,
        gfactors=growpath,
        weights=wtpath,
        adjust_ratios=ratiopath,
        savepath=outdir + 'puf' + str(year) + '.parquet')
    # print('Done saving advanced puf parquet file.')


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

