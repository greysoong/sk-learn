# -*- coding: UTF-8 -*-
"""
file:Supervised6-blobs.py
data:2019-07-1611:27
author:Grey
des:
"""

from sklearn.datasets import make_blobs
from sklearn import svm
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X,y = make_blobs(random_state=42)
linear_svm = svm.LinearSVC().fit(X,y)
#print("Coefficient shape:",linear_svm.coef_.shape)
#print("Intercept shape:",linear_svm.intercept_.shape)
mglearn.plots.plot_2d_classification(linear_svm,X,fill=True,alpha=.7)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
line = np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0',"Class 1","Class 2","Line class 0","Line class 1","Line class2"],loc=(1.01,0.3))
plt.show()