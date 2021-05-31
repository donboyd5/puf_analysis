

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

def get_drops(pdiff_df):
    # develop query to identify bad IRSvariable-stub combinations that we will not target
    # return dataframe of the IRS variable-stub combination
    qxnan = "(abspdiff != abspdiff)"  # hack to identify nan values, query() doesn't allow is.nan()
    qx0 = "(pufvar in @untargeted)"
    qx1 = "(pufvar in @badvars)"
    qx2 = "(common_stub in [1, 2, 3, 4] and pufvar in @bad_stub1_4_vars)"
    qx = qxnan + " or " + qx0 + " or " + qx1 + " or " + qx2

    drops = pdiff_df.query(qx).copy()
    drops = drops.query("common_stub != 0") # we don't need to drop anything in stub 0 because we won't run it
    drops.sort_values(by=['common_stub', 'pufvar'], inplace=True)
    return drops



