# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:18:11 2020

@author: donbo
"""

df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2)

pufsub.info()
vars = ['c00100', 'e00200']
dfsums = pufsub.groupby('common_stub')[vars].sum().reset_index()
dfsums.info()
grand_sums = dfsums[vars].sum().to_frame().transpose() # (name='sum')
grand_sums['common_stub'] = 0
grand_sums.info()
dfsums2 = dfsums.append(.reset_index(), ignore_index=True)
