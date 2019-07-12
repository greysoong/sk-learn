# -*- coding: UTF-8 -*-
"""
file:Supervised.py
data:2019-07-1214:24
author:Grey
des:
"""
import mglearn
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:{}".format(X.shape))
plt.show()