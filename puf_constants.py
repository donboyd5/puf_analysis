# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 04:29:41 2020

@author: donbo
"""

# %% imports
import pandas as pd

# %% filenames and urls
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
HT2_2018 = "18in55cmagi.csv"


# %% agi stubs

IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]

HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]

h2stubs = pd.DataFrame([
    [0, 'All income ranges'],
    [1, 'Under $1'],
    [2, '$1 under $10,000'],
    [3, '$10,000 under $25,000'],
    [4, '$25,000 under $50,000'],
    [5, '$50,000 under $75,000'],
    [6, '$75,000 under $100,000'],
    [7, '$100,000 under $200,000'],
    [8, '$200,000 under $500,000'],
    [9, '$500,000 under $1,000,000'],
    [10, '$1,000,000 or more']],
    columns=['h2stub', 'h2range'])


# %% target varnames (puf names and HT2 names and my names)
targvars_all = ['nret_all', 'nret_mars1', 'nret_mars2', 'c00100', 'e00300', 'e00600']


# %% states
STATES = [ 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

STATES_DCPROA = STATES + ['DC', 'PR', 'OA']

STATES_DCPROAUS = STATES + ['DC', 'PR', 'OA', 'US']


# %% crosswalks
# This is Peter's xwalk plus mine -- it includes more than we will use
PUFHT2_XWALK = {
    'nret_all': 'N1',  # Total population
    'nret_mars1': 'MARS1',  # Single returns number
    'nret_mars2': 'MARS2',  # Joint returns number
    'c00100': 'A00100',  # AGI amount
    'e00200': 'A00200',  # Salary and wage amount
    'e00200_n': 'N00200',  # Salary and wage number
    'e00300': 'A00300',  # Taxable interest amount
    'e00600': 'A00600',  # Ordinary dividends amount
    'c01000': 'A01000',  # Capital gains amount
    'c01000_n': 'N01000',  # Capital gains number
    # check Social Security
    # e02400 is Total Social Security
    # A02500 is Taxable Social Security benefits amount
    'e02400': 'A02500',  # Social Security total (2400)
    'c04470': 'A04470',  # Itemized deduction amount (0 if standard deduction)
    'c04470_n': 'N04470',  # Itemized deduction number (0 if standard deduction)
    'c17000': 'A17000',  # Medical expenses deducted amount
    'c17000_n': 'N17000',  # Medical expenses deducted number
    'c04800': 'A04800',  # Taxable income amount
    'c04800_n': 'N04800',  # Taxable income number
    'c05800': 'A05800',  # Regular tax before credits amount
    'c05800_n': 'N05800',  # Regular tax before credits amount
    'c09600': 'A09600',  # AMT amount
    'c09600_n': 'N09600',  # AMT number
    'e00700': 'A00700',  # SALT amount
    'e00700_n': 'N00700',  # SALT number
    # check pensions
    # irapentot: IRAs and pensions total e01400 + e01500
    # A01750: Taxable IRA, pensions and annuities amount
    'irapentot': 'A01750',
}

# CAUTION: reverse xwalk relies on having only one keyword per value
HT2PUF_XWALK = {val: kw for kw, val in PUFHT2_XWALK.items()}
# list(HT2PUF_XWALK.keys())


# %% Peter's  crosswalks
# Peter's mappings of puf to historical table 2
# "n1": "N1",  # Total population
# "mars1_n": "MARS1",  # Single returns number
# "mars2_n": "MARS2",  # Joint returns number
# "c00100": "A00100",  # AGI amount
# "e00200": "A00200",  # Salary and wage amount
# "e00200_n": "N00200",  # Salary and wage number
# "c01000": "A01000",  # Capital gains amount
# "c01000_n": "N01000",  # Capital gains number
# "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
# "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
# "c17000": "A17000",  # Medical expenses deducted amount
# "c17000_n": "N17000",  # Medical expenses deducted number
# "c04800": "A04800",  # Taxable income amount
# "c04800_n": "N04800",  # Taxable income number
# "c05800": "A05800",  # Regular tax before credits amount
# "c05800_n": "N05800",  # Regular tax before credits amount
# "c09600": "A09600",  # AMT amount
# "c09600_n": "N09600",  # AMT number
# "e00700": "A00700",  # SALT amount
# "e00700_n": "N00700",  # SALT number

    # Maps PUF variable names to HT2 variable names
# VAR_CROSSWALK = {
#     "n1": "N1",  # Total population
#     "mars1_n": "MARS1",  # Single returns number
#     "mars2_n": "MARS2",  # Joint returns number
#     "c00100": "A00100",  # AGI amount
#     "e00200": "A00200",  # Salary and wage amount
#     "e00200_n": "N00200",  # Salary and wage number
#     "c01000": "A01000",  # Capital gains amount
#     "c01000_n": "N01000",  # Capital gains number
#     "c04470": "A04470",  # Itemized deduction amount (0 if standard deduction)
#     "c04470_n": "N04470",  # Itemized deduction number (0 if standard deduction)
#     "c17000": "A17000",  # Medical expenses deducted amount
#     "c17000_n": "N17000",  # Medical expenses deducted number
#     "c04800": "A04800",  # Taxable income amount
#     "c04800_n": "N04800",  # Taxable income number
#     "c05800": "A05800",  # Regular tax before credits amount
#     "c05800_n": "N05800",  # Regular tax before credits amount
#     "c09600": "A09600",  # AMT amount
#     "c09600_n": "N09600",  # AMT number
#     "e00700": "A00700",  # SALT amount
#     "e00700_n": "N00700",  # SALT number
# }

