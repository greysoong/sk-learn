# -*- coding: UTF-8 -*-
"""
file:Feature-interaction-1.py
data:2019-07-3017:11
author:Grey
des:
"""
import mglearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X,y = mglearn.datasets.make_wave(n_samples=100)

line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)


bins = np.linspace(-3,3,11)
which_bin = np.digitize(X,bins=bins)
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
line_binned = encoder.transform(np.digitize(line,bins=bins))
X_combined = np.hstack([X,X_binned])
print(X_combined.shape)

reg = LinearRegression().fit(X_combined,y)
line_combined = np.hstack([line,line_binned])
plt.plot(line,reg.predict(line_combined),label='linear regression combined')

for bin in bins:
    plt.plot([bin,bin],[-3,3],':',c='k')

plt.legend(loc='best')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:,0],y,'o',c='k')
plt.show()