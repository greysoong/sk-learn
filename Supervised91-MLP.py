# -*- coding: UTF-8 -*-
"""
file:Supervised91-MLP.py
data:2019-07-1916:28
author:Grey
des:
"""
import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-3,3,100)
plt.plot(line,np.tanh(line),label="tanh")
plt.plot(line,np.maximum(line,0),label="relu")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x),tanh(X)")
plt.show()