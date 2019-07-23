# -*- coding: UTF-8 -*-
"""
file:Superviser93-iris.py
data:2019-07-238:21
author:Grey
des:
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


iris = load_iris()

X_train,X_test,y_train,y_test =train_test_split(iris.data,iris.target,random_state=42)
gbrt = GradientBoostingClassifier(learning_rate=0.01,random_state=0)
gbrt.fit(X_train,y_train)

print("Decision function shape:{}".format(gbrt.decision_function(X_test).shape))
print("Decision function\n{}".format(gbrt.decision_function(X_test)[:6,:]))
print("Argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test),axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6]))
print("sums:{}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
print("Argmax of predicted:\n{}".format(np.argmax(gbrt.predict_proba(X_test),axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
print("===")
logreg = LogisticRegression()
named_target = iris.target_names[y_train]
logreg.fit(X_train,named_target)
print("unique classes in training data:{}".format(logreg.classes_))
print("predictions:{}".format(logreg.predict_proba(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test),axis=1)
print("argmax of decision function:{}".format(argmax_dec_func[:10]))
print("argmax combined with classes_:{}".format(logreg.classes_[argmax_dec_func][:10]))