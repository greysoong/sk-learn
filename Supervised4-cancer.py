# -*- coding: UTF-8 -*-
"""
file:Supervised4.py
data:2019-07-1215:53
author:Grey
des:
"""
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
traning_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    traning_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,traning_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy,label="trainig accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


'''
print("cancer.keys():\n{}".format(cancer.keys()))
print("Shape of cancer data:{}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))
'''