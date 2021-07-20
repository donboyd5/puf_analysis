# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 07:40:08 2020

@author: donbo
"""

# %% imports
import sys
from pathlib import Path
import os
import numpy as np

# this is sufficient
# from pathlib import Path
WEIGHTING_DIR = str(Path.home() / 'Documents/python_projects/weighting')
if WEIGHTING_DIR not in sys.path:
    sys.path.append(str(WEIGHTING_DIR))
import src.microweight as mw
import src.make_test_problems as mtp
# not needed:
# sys.path.append('c:/programs_python/')
# sys.path.append('c:/programs_python/weighting/src/')
# import programs_python.weighting.src.microweight as mw
# import weighting.src.microweight as mw
# import microweight as mw

# import os

# %% more
# from ..weighting.src import microweight
print(sys.path)
os.getcwd()

p = mtp.Problem(h=40, s=2, k=3)
p.xmat
