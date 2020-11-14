
import pandas as pd
import puf_constants as pc
import puf_utilities as pu

def ulist(thelist):
    # return unique list with out changing order of original list
    ulist = []
    for x in thelist:
        if x not in ulist:
            ulist.append(x)
    return ulist

def get_possible_targets(targets_fname):
    targets_possible = pd.read_csv(targets_fname)

    target_mappings = targets_possible.drop(labels=['common_stub', 'incrange', 'irs'], axis=1).drop_duplicates()
    target_vars = target_mappings.pufvar.to_list()

    # get names of puf variables for which we will need to create nnz indicator
    innz = target_mappings.pufvar.str.contains('_nnz')
    nnz_vars = target_mappings.pufvar[innz]
    pufvars_to_nnz = nnz_vars.str.rsplit(pat='_', n=1, expand=True)[0].to_list()

    possible_wide = targets_possible.loc[:, ['common_stub', 'pufvar', 'irs']] \
        .pivot(index='common_stub', columns='pufvar', values='irs') \
        .reset_index()
    possible_wide.columns.name = None
    return possible_wide


def prep_puf(puf, target_names):
    puf = puf.copy()

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

    puf['ht2_stub'] = pd.cut(
        puf['c00100'],
        pc.HT2_AGI_STUBS,
        labels=range(1, 11),
        right=False)

    puf['filer'] = pu.filers(puf)

    puf['nret_all'] = 1

    # marital status indicators
    puf['mars1'] = puf.MARS.eq(1)
    puf['mars2'] = puf.MARS.eq(2)
    puf['mars3'] = puf.MARS.eq(3)
    puf['mars4'] = puf.MARS.eq(4)
    puf['mars5'] = puf.MARS.eq(5)

    # create capital gains positive and negative
    for var in pos:
        puf[var + 'pos'] = puf[var] * puf[var].gt(0)

    for var in neg:
        puf[var + 'neg'] = puf[var] * puf[var].lt(0)

    for var in nnz:
        puf[var + '_nnz'] = puf[var].ne(0) * 1

    # safely drop columns we don't want to keep
    idvars = ['pid', 'filer', 'common_stub']
    numvars = ['nret_all', 'mars1', 'mars2', 'mars3', 'mars4']
    keep_vars = idvars + numvars + target_names
    keep_vars = ulist(keep_vars)  # keeps unique names in case there is overlap with idvars

    return puf[keep_vars]
