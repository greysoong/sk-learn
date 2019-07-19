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
from mpl_toolkits.mplot3d import Axes3D,axes3d

X,y = make_blobs(centers=4,random_state=8)
y = y%2

linear_svm = svm.LinearSVC().fit(X,y)
X_new = np.hstack([X,X[:,1:]**2])
linear_svm_3d = svm.LinearSVC().fit(X_new,y)
coef,intercept = linear_svm_3d.coef_.ravel(),linear_svm_3d.intercept_
#print("Coefficient shape:",linear_svm.coef_.shape)
#print("Intercept shape:",linear_svm.intercept_.shape)
#mglearn.plots.plot_2d_classification(linear_svm,X)
#mglearn.discrete_scatter(X[:,0],X[:,1],y)


'''
line = np.linspace(-15,15)
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.xlim(-10,8)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0',"Class 1","Class 2","Line class 0","Line class 1","Line class2"],loc=(1.01,0.3))
'''
figure = plt.figure()
ax = Axes3D(figure,elev=-152,azim=-26)
xx = np.linspace(X_new[:,0].min()-2,X_new[:,0].max()+2,50)
yy = np.linspace(X_new[:,1].min()-2,X_new[:,1].max()+2,50)
XX,YY = np.meshgrid(xx,yy)
#ZZ = (coef[0]*XX+coef[1]*YY+intercept)/-coef[2]
ZZ = YY**2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(),YY.ravel(),ZZ.ravel()])
plt.contourf(XX,YY,dec.reshape(XX.shape),levels=[dec.min(),0,dec.max()],cmap=mglearn.cm2,alpha=0.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature 0")
plt.xlabel("Feature 1")

'''
mask = y == 0
ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=0.3)
ax.scatter(X_new[mask,0],X_new[mask,1],X_new[mask,2],c='b',cmap=mglearn.cm2,s=60)
ax.scatter(X_new[~mask,0],X_new[~mask,1],X_new[~mask,2],c='r',marker='^',cmap=mglearn.cm2,s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1**2")
'''
plt.show()