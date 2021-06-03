
import pandas as pd

import functions_geoweight_puf as gwp
import puf_constants as pc

# %% functions
def get_drops_national(pdiff_df):
    # define any variable-stub combinations to drop via a drops dataframe
    # for definitions see: https://pslmodels.github.io/Tax-Calculator/guide/index.html

    # variables we don't want to target (e.g., taxable income or tax after credits)
                # drop net cap gains - instead we are targeting the pos and neg versions
    untargeted = ['c01000', 'c01000_nnz',
                'c04800', 'c04800_nnz',  # regular taxable income
                'c09200', 'c09200_nnz',  # income tax liability (including othertaxes) after non-refundable credits
                # for our new business-like income variables keep only the positive and negative
                # amounts and drop the _nnz and net values
                'e00900', # 'e00900neg_nnz', 'e00900pos_nnz',
                'e02000',
                # maybe drop the partnership/S corp value
                'taxac_irs', 'taxac_irs_nnz']

    # e02400 is Total social security (OASDI) benefits, we are targeting taxable instead
    badvars = ['c02400', 'c02400_nnz']  # would like to target but values are bad

    # the next vars seem just too far off in irs stubs 1-4 to target in those stubs
    # c17000 Sch A: Medical expenses deducted
    # c19700 Sch A: Charity contributions deducted
    bad_stub1_4_vars = ['c17000', 'c17000_nnz', 'c19700', 'c19700_nnz']

    # develop query to identify bad IRSvariable-stub combinations that we will not target
    # return dataframe of the IRS variable-stub combination
    qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values, query() doesn't allow is.nan()
    qx0 = "(pufvar in @untargeted)"
    qx1 = "(pufvar in @badvars)"
    qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @bad_stub1_4_vars)"
    qx = qxnan + " or " + qx0 + " or " + qx1 + " or " + qx2

    drops = pdiff_df.query(qx).copy()
    drops = drops.query("common_stub != 0") # we don't need to drop anything in stub 0 because we won't run it
    drops = drops.sort_values(by=['common_stub', 'pufvar'])
    return drops

def get_wtdsums(pufsub, weightdf, stubvar='common_stub'):
    idcols = ['pid', 'filer', 'common_stub', 'ht2_stub']
    sumcols = [x for x in pufsub.columns.tolist() if x not in idcols]

    df = pd.merge(pufsub, weightdf[['pid', 'weight']], how='left', on='pid')
    df.update(df.loc[:, sumcols].multiply(df.weight, axis=0))
    dfsums = df.groupby(stubvar)[sumcols].sum().reset_index()
    grand_sums = dfsums[sumcols].sum().to_frame().transpose()
    grand_sums[stubvar] = 0
    dfsums = dfsums.append(grand_sums, ignore_index=True)
    dfsums.sort_values(by=stubvar, axis=0, inplace=True)
    dfsums = dfsums.set_index(stubvar, drop=False)
    return dfsums

def get_pufht2_targets(pufsub, weightdf, ht2sharespath, compstates):
    pufsums_ht2 = get_wtdsums(pufsub, weightdf, stubvar='ht2_stub')
    pufsums_ht2long = pd.melt(pufsums_ht2, id_vars='ht2_stub', var_name='pufvar', value_name='pufsum')
    # collapse ht2 shares to the states we want
    ht2_collapsed = gwp.collapse_ht2(ht2sharespath, compstates)
    # create targets by state and ht2_stub from pufsums and collapsed shares, keeping intersection
    ht2targets = pd.merge(ht2_collapsed, pufsums_ht2long, how='inner', on=['pufvar', 'ht2_stub'])
    ht2targets = pd.merge(ht2targets, pc.irspuf_target_map[['pufvar', 'column_description']], how='left', on='pufvar')
    ht2targets['target'] = ht2targets.pufsum * ht2targets.share
    varorder = ['stgroup', 'ht2_stub', 'pufvar', 'ht2var',
                'pufsum', 'ht2', 'share', 'target', 'column_description', 'ht2description']
    return ht2targets[varorder].sort_values(by=['stgroup', 'ht2_stub', 'pufvar'])







