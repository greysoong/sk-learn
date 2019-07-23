# -*- coding: UTF-8 -*-
"""
file:Supervised4.py
data:2019-07-1215:53
author:Grey
des:
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,export_graphviz
import graphviz
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm  import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler

cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=0)
'''
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train)/std_on_train
X_test_scaled = (X_test - mean_on_train)/std_on_train

tree = DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train)
forest = RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)
gbrt =GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)
mlp = MLPClassifier(max_iter=1000,alpha=1,random_state=0)
mlp.fit(X_train_scaled,y_train)


min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled = (X_train - min_on_training)/range_on_training
#print("Minimun for each feature\n{}".format(X_train_scaled.min(axis=0)))
#print("Maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))
print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))
'''
'''
X_test_scaled = (X_test - min_on_training)/range_on_training
svc = SVC(C=1000)
svc.fit(X_train_scaled,y_train)
'''
svm = SVC(C=100)
svm.fit(X_train,y_train)
print("Test set accuracy:{:.2f}".format(svm.score(X_test,y_test)))

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled,y_train)
print("Scaled test set accuracy:{:.2f}".format(svm.score(X_test_scaled,y_test)))
scaler_std = StandardScaler()
scaler_std.fit(X_train)
X_train_scaled_std = scaler_std.transform(X_train)
X_test_scaled_std = scaler_std.transform(X_test)
svm.fit(X_train_scaled_std,y_train)
print("SVM test accuracy:{:.2f}".format(svm.score(X_test_scaled_std,y_test)))


'''
#print("Accuracy on training set:{:.3f}".format(forest.score(X_train,y_train)))
#print("Accuracy on test set:{:.3f}".format(forest.score(X_test,y_test)))
#print("Accuracy on training set:{:.3f}".format(gbrt.score(X_train,y_train)))
#print("Accuracy on test set:{:.3f}".format(gbrt.score(X_test,y_test)))
#print("Accuracy on training set:{:.3f}".format(svc.score(X_train_scaled,y_train)))
#print("Accuracy on test set:{:.3f}".format(svc.score(X_test_scaled,y_test)))
print("Accuracy on training set:{:.3f}".format(mlp.score(X_train_scaled,y_train)))
print("Accuracy on test set:{:.3f}".format(mlp.score(X_test_scaled,y_test)))

plt.figure(figsize=(20,5))
plt.imshow(mlp.coefs_[0],interpolation="none",cmap="viridis")
plt.yticks(range(30),cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
'''

'''
plt.plot(X_train.min(axis=0),'o',label="min")
plt.plot(X_train.max(axis=0),'^',label="max")
plt.legend(loc=4)
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
#plot_feature_importances_cancer(gbrt)
plt.show()
'''


'''
print("Accuracy on training set:{:.3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(tree.score(X_test,y_test)))
export_graphviz(tree,out_file="tree.dot",class_names=["malignant","benign"],feature_names=cancer.feature_names,impurity=False,filled=True)
print("Feature importance:\n{}".format(tree.feature_importances_))
'''

'''
for C,marker in zip([0.001,1,100],['o','^','v']):
    lr_l1 = LogisticRegression(C=C,penalty='l1').fit(X_train,y_train)
    print("Training accuracy of l1 logreg with C={:.3f}:{:.2f}".format(C,lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}:{:.2f}".format(C, lr_l1.score(X_test, y_test)))

    plt.plot(lr_l1.coef_.T,marker,label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
    plt.hlines(0,0,cancer.data.shape[1])
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-5,5)
    plt.legend(loc=3)
plt.show()
'''

'''

logreg = LogisticRegression().fit(X_train,y_train)
print("Training set score:{:.3f}".format(logreg.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg.score(X_test,y_test)))
logreg100 = LogisticRegression(C=100).fit(X_train,y_train)
print("Training set score:{:.3f}".format(logreg100.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg100.score(X_test,y_test)))
logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
print("Training set score:{:.3f}".format(logreg001.score(X_train,y_train)))
print("Test set score:{:.3f}".format(logreg001.score(X_test,y_test)))
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.plot(logreg100.coef_.T,'o',label="C=100")
plt.plot(logreg001.coef_.T,'o',label="C=0.01")
plt.xticks(range(cancer.data.shape[1]),cancer.feature_names,rotation=90)
plt.hlines(0,0,cancer.data.shape[1])
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()
'''
'''
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

'''
print("cancer.keys():\n{}".format(cancer.keys()))
print("Shape of cancer data:{}".format(cancer.data.shape))
print("Sample counts per class:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))
'''