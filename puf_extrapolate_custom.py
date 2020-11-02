# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:27:39 2020

@author: donbo
"""

import numpy as np


def extrapolate_custom(puf, gf_custom, year=2017):
    """
    Apply to variables the grow factor values for specified calendar year.

    gfcustom has CUMULATIVE growth factors to a year, not values for a year

    growfactors differences from tax-calculator:
        ADIVS_ADJ use instead of ADIVS for e00600 and e00650
        ACHARITY use instead of ATXPY for e19800 and e20100
        ASALT use instead of ATXPY for e18400, e18500
    """
    # gfv.at[year - 2011, 'ASALT']
    # for name in GrowFactors.VALID_NAMES:
    #     gfv[name] = pufx.gfactors.factor_value(name, year)

    # some differences from tax-calculator's _extrapolate
    #    its function advances one year at a time
    #    this function advances all at once
    #    thus we need to calculate the cumulative growth factors to 2017
    #    and use those

    # apply values to Records variables

    pufx = puf.copy()

    # construct cumulative growfactors
    gfv = gf_custom.copy()
    vars = gfv.columns.tolist()
    vars.remove('YEAR')
    gfv[vars] = gfv[vars].cumprod()

    # MAIN INCOME COMPONENTS
    pufx.e00200 *= gfv.at[year - 2011, 'AWAGE']
    pufx.e00200p *= gfv.at[year - 2011, 'AWAGE']
    pufx.e00200s *= gfv.at[year - 2011, 'AWAGE']
    pufx.pencon_p *= gfv.at[year - 2011, 'AWAGE']
    pufx.pencon_s *= gfv.at[year - 2011, 'AWAGE']
    pufx.e00300 *= gfv.at[year - 2011, 'AINTS']
    pufx.e00400 *= gfv.at[year - 2011, 'AINTS']
    pufx.e00600 *= gfv.at[year - 2011, 'ADIVS_ADJ']  # changed from ADIVS
    pufx.e00650 *= gfv.at[year - 2011, 'ADIVS_ADJ']  # changed from ADIVS
    pufx.e00700 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e00800 *= gfv.at[year - 2011, 'ATXPY']

    pufx.e00900s = np.where(pufx.e00900s >= 0,
                            pufx.e00900s * gfv.at[year - 2011, 'ASCHCI'],
                            pufx.e00900s * gfv.at[year - 2011, 'ASCHCL'])
    pufx.e00900p = np.where(pufx.e00900p >= 0,
                            pufx.e00900p * gfv.at[year - 2011, 'ASCHCI'],
                            pufx.e00900p * gfv.at[year - 2011, 'ASCHCL'])
    pufx.e00900 = pufx.e00900p + pufx.e00900s

    # original lines below generated slice warnings; I removed [:]
    # pufx.e00900s[:] = np.where(pufx.e00900s >= 0,
    #                         pufx.e00900s * gfv.at[year - 2011, 'ASCHCI'],
    #                         pufx.e00900s * gfv.at[year - 2011, 'ASCHCL'])
    # pufx.e00900p[:] = np.where(pufx.e00900p >= 0,
    #                         pufx.e00900p * gfv.at[year - 2011, 'ASCHCI'],
    #                         pufx.e00900p * gfv.at[year - 2011, 'ASCHCL'])
    # pufx.e00900[:] = pufx.e00900p + pufx.e00900s

    pufx.e01100 *= gfv.at[year - 2011, 'ACGNS']
    pufx.e01200 *= gfv.at[year - 2011, 'ACGNS']
    pufx.e01400 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e01500 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e01700 *= gfv.at[year - 2011, 'ATXPY']

    pufx.e02000 = np.where(pufx.e02000 >= 0,
                           pufx.e02000 * gfv.at[year - 2011, 'ASCHEI'],
                           pufx.e02000 * gfv.at[year - 2011, 'ASCHEL'])

    # original lines below generated slice warning; I removed [:]
    # pufx.e02000[:] = np.where(pufx.e02000 >= 0,
    #                           pufx.e02000 * gfv.at[year - 2011, 'ASCHEI'],
    #                           pufx.e02000 * gfv.at[year - 2011, 'ASCHEL'])


    pufx.e02100 *= gfv.at[year - 2011, 'ASCHF']
    pufx.e02100p *= gfv.at[year - 2011, 'ASCHF']
    pufx.e02100s *= gfv.at[year - 2011, 'ASCHF']
    pufx.e02300 *= gfv.at[year - 2011, 'AUCOMP']
    pufx.e02400 *= gfv.at[year - 2011, 'ASOCSEC']
    pufx.e03150 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03210 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03220 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03230 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03270 *= gfv.at[year - 2011, 'ACPIM']
    pufx.e03240 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03290 *= gfv.at[year - 2011, 'ACPIM']
    pufx.e03300 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03400 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e03500 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e07240 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e07260 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e07300 *= gfv.at[year - 2011, 'ABOOK']
    pufx.e07400 *= gfv.at[year - 2011, 'ABOOK']
    pufx.p08000 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e09700 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e09800 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e09900 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e11200 *= gfv.at[year - 2011, 'ATXPY']

    # ITEMIZED DEDUCTIONS
    pufx.e17500 *= gfv.at[year - 2011, 'ACPIM']
    pufx.e18400 *= gfv.at[year - 2011, 'ASALT']  # changed from ATXPY
    pufx.e18500 *= gfv.at[year - 2011, 'ASALT']  # changed from ATXPY
    pufx.e19200 *= gfv.at[year - 2011, 'AIPD']
    pufx.e19800 *= gfv.at[year - 2011, 'ACHARITY']  # changed from ATXPY
    pufx.e20100 *= gfv.at[year - 2011, 'ACHARITY']  # changed from ATXPY
    pufx.e20400 *= gfv.at[year - 2011, 'ATXPY']
    pufx.g20500 *= gfv.at[year - 2011, 'ATXPY']

    # CAPITAL GAINS
    pufx.p22250 *= gfv.at[year - 2011, 'ACGNS']
    pufx.p23250 *= gfv.at[year - 2011, 'ACGNS']
    pufx.e24515 *= gfv.at[year - 2011, 'ACGNS']
    pufx.e24518 *= gfv.at[year - 2011, 'ACGNS']

    # SCHEDULE E
    pufx.e26270 *= gfv.at[year - 2011, 'ASCHEI']
    pufx.e27200 *= gfv.at[year - 2011, 'ASCHEI']
    pufx.k1bx14p *= gfv.at[year - 2011, 'ASCHEI']
    pufx.k1bx14s *= gfv.at[year - 2011, 'ASCHEI']

    # MISCELLANOUS SCHEDULES
    pufx.e07600 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e32800 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e58990 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e62900 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e87530 *= gfv.at[year - 2011, 'ATXPY']
    pufx.e87521 *= gfv.at[year - 2011, 'ATXPY']
    pufx.cmbtp *= gfv.at[year - 2011, 'ATXPY']

    # BENEFITS djb I had to comment these out as they are only available in cps
    # pufx.other_ben *= gfv.at[year - 2011, 'ABENOTHER']
    # pufx.mcare_ben *= gfv.at[year - 2011, 'ABENMCARE']
    # pufx.mcaid_ben *= gfv.at[year - 2011, 'ABENMCAID']
    # pufx.ssi_ben *= gfv.at[year - 2011, 'ABENSSI']
    # pufx.snap_ben *= gfv.at[year - 2011, 'ABENSNAP']
    # pufx.wic_ben *= gfv.at[year - 2011, 'ABENWIC']
    # pufx.housing_ben *= gfv.at[year - 2011, 'ABENHOUSING']
    # pufx.tanf_ben *= gfv.at[year - 2011, 'ABENTANF']
    # pufx.vet_ben *= gfv.at[year - 2011, 'ABENVET']

    return pufx


def extrapolate_custom_bak(puf, gfcustom, year=2017):
    """
    Apply to variables the grow factor values for specified calendar year.

    gfcustom has CUMULATIVE growth factors to a year, not values for a year

    growfactors differences from tax-calculator:
        ADIVS_ADJ use instead of ADIVS for e00600 and e00650
        ACHARITY use instead of ATXPY for e19800 and e20100
        ASALT use instead of ATXPY for e18400, e18500
    """
    gfv = gfcustom
    # gfv.at[year - 2011, 'ASALT']
    # for name in GrowFactors.VALID_NAMES:
    #     gfv[name] = puf.gfactors.factor_value(name, year)

    # some differences from tax-calculator's _extrapolate
    #    its function advances one year at a time
    #    this function advances all at once
    #    thus we need to calculate the cumulative growth factors to 2017
    #    and use those

    # apply values to Records variables

    # MAIN INCOME COMPONENTS
    puf.e00200 *= gfv['AWAGE']
    puf.e00200p *= gfv['AWAGE']
    puf.e00200s *= gfv['AWAGE']
    puf.pencon_p *= gfv['AWAGE']
    puf.pencon_s *= gfv['AWAGE']
    puf.e00300 *= gfv['AINTS']
    puf.e00400 *= gfv['AINTS']
    puf.e00600 *= gfv['ADIVS_ADJ']  # changed from ADIVS
    puf.e00650 *= gfv['ADIVS_ADJ']  # changed from ADIVS
    puf.e00700 *= gfv['ATXPY']
    puf.e00800 *= gfv['ATXPY']
    puf.e00900s[:] = np.where(puf.e00900s >= 0,
                              puf.e00900s * gfv['ASCHCI'],
                              puf.e00900s * gfv['ASCHCL'])
    puf.e00900p[:] = np.where(puf.e00900p >= 0,
                              puf.e00900p * gfv['ASCHCI'],
                              puf.e00900p * gfv['ASCHCL'])
    puf.e00900[:] = puf.e00900p + puf.e00900s
    puf.e01100 *= gfv['ACGNS']
    puf.e01200 *= gfv['ACGNS']
    puf.e01400 *= gfv['ATXPY']
    puf.e01500 *= gfv['ATXPY']
    puf.e01700 *= gfv['ATXPY']
    puf.e02000[:] = np.where(puf.e02000 >= 0,
                             puf.e02000 * gfv['ASCHEI'],
                             puf.e02000 * gfv['ASCHEL'])
    puf.e02100 *= gfv['ASCHF']
    puf.e02100p *= gfv['ASCHF']
    puf.e02100s *= gfv['ASCHF']
    puf.e02300 *= gfv['AUCOMP']
    puf.e02400 *= gfv['ASOCSEC']
    puf.e03150 *= gfv['ATXPY']
    puf.e03210 *= gfv['ATXPY']
    puf.e03220 *= gfv['ATXPY']
    puf.e03230 *= gfv['ATXPY']
    puf.e03270 *= gfv['ACPIM']
    puf.e03240 *= gfv['ATXPY']
    puf.e03290 *= gfv['ACPIM']
    puf.e03300 *= gfv['ATXPY']
    puf.e03400 *= gfv['ATXPY']
    puf.e03500 *= gfv['ATXPY']
    puf.e07240 *= gfv['ATXPY']
    puf.e07260 *= gfv['ATXPY']
    puf.e07300 *= gfv['ABOOK']
    puf.e07400 *= gfv['ABOOK']
    puf.p08000 *= gfv['ATXPY']
    puf.e09700 *= gfv['ATXPY']
    puf.e09800 *= gfv['ATXPY']
    puf.e09900 *= gfv['ATXPY']
    puf.e11200 *= gfv['ATXPY']

    # ITEMIZED DEDUCTIONS
    puf.e17500 *= gfv['ACPIM']
    puf.e18400 *= gfv['ASALT']  # changed from ATXPY
    puf.e18500 *= gfv['ASALT']  # changed from ATXPY
    puf.e19200 *= gfv['AIPD']
    puf.e19800 *= gfv['ACHARITY']  # changed from ATXPY
    puf.e20100 *= gfv['ACHARITY']  # changed from ATXPY
    puf.e20400 *= gfv['ATXPY']
    puf.g20500 *= gfv['ATXPY']

    # CAPITAL GAINS
    puf.p22250 *= gfv['ACGNS']
    puf.p23250 *= gfv['ACGNS']
    puf.e24515 *= gfv['ACGNS']
    puf.e24518 *= gfv['ACGNS']

    # SCHEDULE E
    puf.e26270 *= gfv['ASCHEI']
    puf.e27200 *= gfv['ASCHEI']
    puf.k1bx14p *= gfv['ASCHEI']
    puf.k1bx14s *= gfv['ASCHEI']

    # MISCELLANOUS SCHEDULES
    puf.e07600 *= gfv['ATXPY']
    puf.e32800 *= gfv['ATXPY']
    puf.e58990 *= gfv['ATXPY']
    puf.e62900 *= gfv['ATXPY']
    puf.e87530 *= gfv['ATXPY']
    puf.e87521 *= gfv['ATXPY']
    puf.cmbtp *= gfv['ATXPY']

    # BENEFITS
    puf.other_ben *= gfv['ABENOTHER']
    puf.mcare_ben *= gfv['ABENMCARE']
    puf.mcaid_ben *= gfv['ABENMCAID']
    puf.ssi_ben *= gfv['ABENSSI']
    puf.snap_ben *= gfv['ABENSNAP']
    puf.wic_ben *= gfv['ABENWIC']
    puf.housing_ben *= gfv['ABENHOUSING']
    puf.tanf_ben *= gfv['ABENTANF']
    puf.vet_ben *= gfv['ABENVET']

    # remove local dictionary
    # del gfv
    return puf

