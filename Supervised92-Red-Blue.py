# -*- coding: UTF-8 -*-
"""
file:Supervised92-Red-Blue.py
data:2019-07-2215:15
author:Grey
des:
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
X,y = make_circles(noise=0.25,factor=0.5,random_state=1)

y_named = np.array(["blue","red"])[y]

X_train,X_test,y_train_named,y_test_named,y_train,y_test =train_test_split(X,y_named,y,random_state=0)
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train_named)

print("X_test.shape:{}".format(X_test.shape))
print("Decision function shape:{}".format(gbrt.decision_function(X_test).shape))
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6]))
print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test)>0))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
greater_zero = (gbrt.decision_function(X_test)>0).astype(int)
pred = gbrt.classes_[greater_zero]
print("pred is equal to predictions:{}".format(np.all(pred == gbrt.predict(X_test))))
decision_function = gbrt.decision_function(X_test)
print("Decision function minimum:{:.2f}maximum:{:.2f}".format(np.min(decision_function),np.max(decision_function)))
print("Shape of probabilities:{}".format(gbrt.predict_proba(X_test).shape))
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))


fig,axes = plt.subplots(1,2,figsize = (13,5))
mglearn.tools.plot_2d_separator(gbrt,X,ax=axes[0],alpha=.4,fill=True,cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt,X,ax=axes[1],alpha=.4,cm=mglearn.ReBl,function='predict_proba')

for ax in axes:
    mglearn.discrete_scatter(X_test[:,0],X_test[:,1],y_test,markers='^',ax=ax)
    mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train,markers='o',ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

cbar = plt.colorbar(scores_image)
axes[0].legend(["Test class 0","Test class 1","Train class 0","Train class 1"],ncol=4,loc=(.1,1.1))

plt.show()
