# -*- coding: UTF-8 -*-
"""
file:Supervised5-boston.py
data:2019-07-1216:08
author:Grey
des:
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import numpy as np

boston = load_boston()
#print("Data shape:{}".format(boston.data.shape))
X,y = mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test =train_test_split(X,y,random_state=0)
#print("X.shape:{}".format((X.shape)))
lr = LinearRegression().fit(X_train,y_train)
ridge = Ridge().fit(X_train,y_train)
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)

'''
print("Training set score:{:.2f}".format(lr.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lr.score(X_test,y_test)))
print("Training set score:{:.2f}".format(ridge.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge.score(X_test,y_test)))
print("Training set score:{:.2f}".format(ridge10.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge10.score(X_test,y_test)))
print("Training set score:{:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score:{:.2f}".format(ridge01.score(X_test,y_test)))
'''
lasso = Lasso().fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso.coef_ !=0)))
lasso001 = Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso001.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso001.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso001.coef_ !=0)))
lasso00001 = Lasso(alpha=0.0001,max_iter=100000).fit(X_train,y_train)
print("Training set score:{:.2f}".format(lasso00001.score(X_train,y_train)))
print("Test set score:{:.2f}".format(lasso00001.score(X_test,y_test)))
print("Number of features used:{}".format(np.sum(lasso00001.coef_ !=0)))
plt.plot(lasso.coef_,'s',label="Ridge alpha=1")
plt.plot(lasso001.coef_,'^',label="Ridge alpha=0.01")
plt.plot(lasso00001.coef_,'v',label="Ridge alpha=0.0001")
plt.plot(ridge01.coef_,'o',label='Ridge alpha=0.1')
plt.xlabel("Coefficient index")
plt.ylabel("coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend(ncol=1,loc=(0,1.05))
plt.show()

"""
plt.plot(ridge.coef_,'s',label="Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label="Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label="Ridge alpha=0.1")

plt.plot(lr.coef_,'o',label='LinearRegression')
plt.xlabel("Coefficient index")
plt.ylabel("coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()
"""