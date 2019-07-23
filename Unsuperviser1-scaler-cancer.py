# -*- coding: UTF-8 -*-
"""
file:Unsuperviser1-scaler.py
data:2019-07-239:56
author:Grey
des:
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import mglearn

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test =train_test_split(cancer.data,cancer.target,random_state=1)
'''
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
#print("transformed shape:{}".format(X_train_scaled.shape))
#print("per-feature minimum before scaling:\n{}".format(X_train.min(axis=0)))
#print("per-feature maximum before scaling:\n{}".format(X_train.max(axis=0)))
#print("per-feature minimum after scaling:\n{}".format(X_train_scaled.min(axis=0)))
#print("per-feature maximum after scaling:\n{}".format(X_train_scaled.max(axis=0)))
X_test_scaled = scaler.transform(X_test)
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
'''
fig,axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()

for i in range(30):
    _,bins = np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant","benign"],loc="best")
fig.tight_layout()
plt.show()
'''