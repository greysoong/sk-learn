# -*- coding: UTF-8 -*-
"""
file:Supervised3.py
data:2019-07-1215:36
author:Grey
des:
"""
import mglearn
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()