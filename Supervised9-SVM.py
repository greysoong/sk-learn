# -*- coding: UTF-8 -*-
"""
file:Supervised9-SVM.py
data:2019-07-199:23
author:Grey
des:
"""
from sklearn.svm import SVC
import mglearn
import matplotlib.pyplot as plt

X,y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf',C=10,gamma=0.1).fit(X,y)
mglearn.plots.plot_2d_separator(svm,X,eps=.5)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0],sv[:,1],sv_labels,s=15,markeredgewidth=3)
fig,axes = plt.subplots(3,3,figsize=(15,10))

for ax,C in zip (axes,[-1,0,3]):
    for a , gamma in zip(ax,range(-1,2)):
        mglearn.plots.plot_svm(log_C=C,log_gamma=gamma,ax=a)


plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
axes[0,0].legend(["Class 0","class1","sv class 0","sv class 1"],ncol=4,loc=(.9,1.2))
plt.show()