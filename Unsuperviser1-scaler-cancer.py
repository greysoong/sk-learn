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


cancer = load_breast_cancer()
X_train,X_test,y_train,y_test =train_test_split(cancer.data,cancer.target,random_state=1)

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

