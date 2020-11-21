# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 06:53:41 2020

@author: donbo
"""

temp = pd.read_csv(HT2_SHARES)
pu.uvals(temp.pufvar)

temp.pufvar.unique()
ht2 = pd.read_csv(ht2_path)  # 87,450

temp = ht2[['ht2var', 'ht2description', 'pufvar']].drop_duplicates()
temp.to_csv(r'c:\temp\temp.csv')
