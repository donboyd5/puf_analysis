# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 03:14:39 2020

@author: donbo
"""

# %% notes
# https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html

# https://plotly.com/python/table/
# https://mode.com/example-gallery/python_dataframe_styling/



# %% imports
import plotly.graph_objects as go
import pandas as pd


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
