# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 09:24:24 2020

@author: donbo
"""


a = np.array([1,1,2,3,4,5,6,7])
b = np.array([1,1,2,3,4,5,6,7])
b = np.array([1,1,2,3,4,5,6,7])

c = a + b
d = a
d = a + b
d[:] = a + b

a
b
type(a)
type(a[:])

a[:] = np.where(a > 2, a * 10, a *2)
a

b = np.where(b > 2, b * 10, b *2)
b



#  pufx.e00900[:] = pufx.e00900p + pufx.e00900s

    # pufx.e00900p[:] = np.where(pufx.e00900p >= 0,
    #                         pufx.e00900p * gfv.at[year - 2011, 'ASCHCI'],
    #                         pufx.e00900p * gfv.at[year - 2011, 'ASCHCL'])

d = np.where(b > 1)

