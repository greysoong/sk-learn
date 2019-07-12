# -*- coding: UTF-8 -*-
"""
file:Supervised.py
data:2019-07-1214:24
author:Grey
des:
"""
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("Test set predictions:{}".format(clf.predict(X_test)))
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))
'''
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0","Class 1"],loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:{}".format(X.shape))
plt.show()
'''