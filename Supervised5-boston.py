# -*- coding: UTF-8 -*-
"""
file:Supervised5-boston.py
data:2019-07-1216:08
author:Grey
des:
"""
from sklearn.datasets import load_boston
import mglearn

boston = load_boston()
print("Data shape:{}".format(boston.data.shape))
X,y = mglearn.datasets.load_extended_boston()
print("X.shape:{}".format((X.shape)))