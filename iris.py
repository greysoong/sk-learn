# -*- coding: UTF-8 -*-
"""
file:iris.py
data:2019-07-119:23
author:Grey
des:
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas.plotting
iris_dataset = load_iris()


'''
print ("Keys of iris_dataset:\n{}".format(iris_dataset.keys()))
print (iris_dataset['DESCR'][:193]+"\n...")
print ("Target names:{}".format(iris_dataset['target_names']))
print ("Feature names:\n{}".format(iris_dataset['feature_names']))
print ('Type of data:{}'.format(type(iris_dataset['data'])))
print ("Shape of data:{}".format(iris_dataset['data'].shape))
print ('First five rows of data :\n{}'.format(iris_dataset['data'][:5]))
print ("Type of target:{}".format(type(iris_dataset['target'])))
print ("Shape of target:{}".format(iris_dataset['target'].shape))
print ("Target:\n{}".format(iris_dataset['target']))
'''
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
'''
print ("X_train shape:{}".format(X_train.shape))
print ("y_train shape:{}".format(y_train.shape))
print ("X_test shape:{}".format(X_test.shape))
print ("y_test shape:{}".format(y_test.shape))
'''
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr = pd