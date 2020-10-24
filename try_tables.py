# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 03:14:39 2020

@author: donbo
"""

# %% notes
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
# https://pbpython.com/styling-pandas.html
# https://mkaz.blog/code/python-string-format-cookbook/
# https://www.youtube.com/watch?v=Sj42rqym9lk
# https://mode.com/example-gallery/python_dataframe_styling/

# https://github.com/spyder-ide/spyder-notebook
# https://groups.google.com/g/spyderlib

# https://plotly.com/python/table/
# https://mode.com/example-gallery/python_dataframe_styling/



# %% imports
import pandas as pd

import plotly.graph_objects as go
import matplotlib


# %% check 1
df = pd.DataFrame([[3,2,10,4],[20,1,3,2],[5,4,6,1]])
df.style.background_gradient()

pathfn = r'c:\temp\fn.html'
# x create and write to new file
# w for writing, overwrite existing
# a for appending
# note that directory must exist
f = open(pathfn, mode='a')  
f.write(df.style.background_gradient().render())
f.close()


# %% data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color='lavender',
               align='left'))
])

fig.show()






