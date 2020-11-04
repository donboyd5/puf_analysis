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

