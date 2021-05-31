
import numpy as np
import pandas as pd
# from functools import reduce

import puf_constants as pc
import puf_utilities as pu

def comp_report(pdiff_df, outfile, title, ipdiff_df=None):

    # comparison report
    #   pdiff_df, and ipdiff_df if present, are data frames created by rwp.get_pctdiffs

    print(f'Preparing report...')
    comp = pdiff_df.copy()
    if ipdiff_df is not None:
        keep = ['common_stub', 'pufvar', 'pdiff']
        ipdiff_df.loc[:, keep].rename(columns={'pdiff': 'ipdiff'})
        comp = pd.merge(comp, ipdiff_df.loc[:, keep].rename(columns={'pdiff': 'ipdiff'}), how='left', on=['common_stub', 'pufvar'])

    # we already have column_description so don't bring it in again
    comp = pd.merge(comp, pc.irspuf_target_map.drop(columns=['column_description']), how='left', on='pufvar')  # get irsvar and other documentation


    ordered_vars = ['common_stub', 'incrange', 'pufvar', 'irsvar',
                    'target', 'puf', 'diff',
                    'pdiff', 'ipdiff', 'column_description']  # drop abspdiff
    if ipdiff_df is None: ordered_vars.remove('ipdiff')
    comp = comp[ordered_vars]

    # sort by pufvar dictionary order (pd.Categorical)
    comp['pufvar'] = pd.Categorical(comp.pufvar,
                                    categories=pc.pufirs_fullmap.keys(),
                                    ordered=True)

    comp.sort_values(by=['pufvar', 'common_stub'], axis=0, inplace=True)

    target_vars = comp.pufvar.unique()

    print(f'Writing report...')
    s = comp.copy()

    s['pdiff'] = s['pdiff'] / 100.0
    if ipdiff_df is not None: s['ipdiff'] = s['ipdiff'] / 100.0
    format_mapping = {'target': '{:,.0f}',
                      'puf': '{:,.0f}',
                      'diff': '{:,.0f}',
                      'pdiff': '{:.1%}',
                      'ipdiff': '{:.1%}'}
    if ipdiff_df is None: format_mapping.pop('ipdiff')
    for key, value in format_mapping.items():
        s[key] = s[key].apply(value.format)

    tfile = open(outfile, 'a')
    tfile.truncate(0)
    # first write a summary with stub 0 for all variables
    tfile.write('\n' + title + '\n\n')
    tfile.write('Data are for filers only, using IRS filing requirements plus estimates of likely filing.\n')
    tfile.write("  Columns 'diff' and 'pdiff' give the puf value minus target, and diff as % of target, respectively.\n")
    tfile.write("  If column 'ipdiff' is present, it gives the % difference between puf and target using a different set of weights that should be mentioned in report title.\n")
    tfile.write('\nThis report is in 3 sections:\n')
    tfile.write('  1. Summary report for all variables, summarized over all filers\n')
    tfile.write('  2. Detailed report by AGI range for each variable\n')
    tfile.write('  3. Table that provides details on puf variables and their mappings to irs data\n')
    tfile.write('\n1. Summary report for all variables, summarized over all filers:\n\n')
    s2 = s[s.common_stub==0]
    tfile.write(s2.to_string())

    # now write details for each variable
    tfile.write('\n\n2. Detailed report by AGI range for each variable:')
    for var in target_vars:
        tfile.write('\n\n')
        s2 = s[s.pufvar==var]
        tfile.write(s2.to_string())

    # finally, write the mapping
    tfile.write('\n\n\n3. Detailed report on variable mappings\n\n')
    tfile.write(pc.irspuf_target_map.to_string())
    tfile.close()
    print("All done.")

    return #  comp return nothing or return comp?


# class Compreport:
#     """Class with
#     """

    # def __init__(self, wh, xmat, targets=None, geotargets=None):
    #     self.wh = wh
    #     self.xmat = xmat
    #     self.targets = targets
    #     self.geotargets = geotargets
    #     self.targets_init = np.dot(self.xmat.T, self.wh)
    #     if self.targets is not None:
    #         self.pdiff_init = self.targets_init / self.targets * 100 - 100

    # def reweight(self,
    #              method='ipopt',
    #              options=None):
        # here are the results we want for every method
        # fields = ('method',
        #           'elapsed_seconds',
        #           'sspd',
        #           'wh_opt',
        #           'targets_opt',
        #           'pdiff',
        #           'g',
        #           'opts',
        #           'method_result')
        # ReweightResult = namedtuple('ReweightResult', fields, defaults=(None,) * len(fields))

        # rwres = ReweightResult(method=method,
        #                        elapsed_seconds=method_result.elapsed_seconds,
        #                        sspd=sspd,
        #                        wh_opt=method_result.wh_opt,
        #                        targets_opt=method_result.targets_opt,
        #                        pdiff=pdiff,
        #                        g=method_result.g,
        #                        opts=method_result.opts,
        #                        method_result=method_result)

        # return rwres

