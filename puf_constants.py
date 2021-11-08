# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 04:29:41 2020

@author: donbo
"""

# %% imports
import pandas as pd
import json
from pathlib import Path


# %%  locations
machine = 'windows'
# machine = 'linux'

if machine == 'windows':
    DIR_FOR_OFFICIAL_PUF = r'C:\Users\donbo\Dropbox (Personal)\PUF files\files_based_on_puf2011/2020-08-20/'
    # DATADIR = Path(r'C:\programs_python\puf_analysis\data/')
    # DATADIR = Path(r'C:\Users\donbo\Documents\python_projects\puf_analysis\data/')
    DATADIR = Path(r'C:\Users\donbo\Dropbox\puf_analysis_materials_from_linux\ignore\data/')
    # the following locations store files not saved in git
    IGNOREDIR = r'C:\programs_python\puf_analysis\ignore/'
elif machine == 'linux':
    # /home/donboyd/Dropbox/PUF files/files_based_on_puf2011
    DIR_FOR_OFFICIAL_PUF = Path(r'~/Dropbox/PUF files/files_based_on_puf2011/2020-08-20/')
    DATADIR = Path('/media/don/ignore/data/')
    IGNOREDIR = Path('/media/don/ignore/')

# print(DATADIR)


# %% filenames and urls
# https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
WEBDIR = 'https://www.irs.gov/pub/irs-soi/'
HT2_2017 = "17in55cmagi.csv"
HT2_2018 = "18in55cmagi.csv"

# DATADIR = r'C:\programs_python\puf_analysis\data/'
# DATADIR = Path.cwd() / 'data'
TARGET_MAP = DATADIR / 'target_mappings.csv'


# %% agi stubs

IRS_AGI_STUBS = [-9e99, 1.0, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]

# common stubs is the common grouping between two different IRS stubs
COMMON_STUBS = [-9e99, 5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 40e3, 50e3,
                 75e3, 100e3, 200e3, 500e3, 1e6, 1.5e6, 2e6, 5e6, 10e6, 9e99]


HT2_AGI_STUBS = [-9e99, 1.0, 10e3, 25e3, 50e3, 75e3, 100e3,
                 200e3, 500e3, 1e6, 9e99]

ht2stubs = pd.DataFrame([
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
    columns=['ht2stub', 'ht2range'])

# this next set of stubs is common between the ht2 stubs and the
# national stubs that are common between itemizers and nonitemizers
ht2common_stubs = pd.DataFrame([
    [0, 'All returns'],
    [1, 'Under $10,000'],
    [2, '$10,000 under $25,000'],
    [3, '$25,000 under $50,000'],
    [4, '$50,000 under $75,000'],
    [5, '$75,000 under $100,000'],
    [6, '$100,000 under $200,000'],
    [7, '$200,000 under $500,000'],
    [8, '$500,000 under $1,000,000'],
    [9, '$1,000,000 or more']],
    columns=['ht2common_stub', 'incrange'])


# use the common IRS stubs, which are the LCD of the main irsstubs and the
# itemized deduction IRS stubs
# irsstubs = pd.read_csv(DATADIR + 'irsstub_labels.csv')
irsstubs = pd.read_csv(DATADIR / 'irsstub_common_labels.csv')

irspuf_target_map = pd.read_csv(TARGET_MAP)

# get previously determined national-puf mapping
# json.dump(pufirs_fullmap, open(DATADIR + 'pufirs_fullmap.json', 'w'))
# this dict defines the order in which we want tables sorted, so get it
pufirs_fullmap = json.load(open(DATADIR / 'pufirs_fullmap.json'))

ht2puf_fullmap = json.load(open(DATADIR / 'ht2puf_fullmap.json'))

pufvars = pd.read_csv(DATADIR / 'pufvars.csv')

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
    'e00700': 'A00700',  # SALT amount djb NO NO NO
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

