# -*- coding: UTF-8 -*-
"""
file:Unsuperviser3-nmf-people1.py
data:2019-07-2416:16
author:Grey
des:
"""
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,NMF
import mglearn

people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape,dtype = np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people/255

X_train,X_test,y_train,y_test = train_test_split(X_people,y_people,stratify=y_people,random_state=0)

'''
nmf = NMF(n_components=15,random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
'''

'''
fix,axes = plt.subplots(3,5,figsize = (15,12),subplot_kw={'xticks':(),'yticks':()})

for i,(component,ax) in enumerate(zip(nmf.components_,axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}.component".format(i))
'''
'''
compn = 3
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig,axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),"yticks":()})
for i ,(ind,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))

compn = 7
inds = np.argsort(X_train_nmf[:,compn])[::-1]
fig,axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),"yticks":()})
for i ,(ind,ax) in enumerate(zip(inds,axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
    
'''
S = mglearn.datasets.make_signals()
A = np.random.RandomState(0).uniform(size=(100,3))
X = np.dot(S,A.T)
print("Shape of measurements:{}".format(X.shape))
nmf = NMF(n_components=3,random_state=42)
S_ = nmf.fit_transform(X)
#print ("Recovered signal shape:{}".format(S_.shape))
pca= PCA(n_components=3)
H = pca.fit_transform(X)
models = [X,S,S_,H]
names = ['Observations(first three measurements)','True sources','NMF recovered signals','PCA recovered signals']

fig,axes = plt.subplots(4,figsize=(8,4),gridspec_kw={'hspace': .5},subplot_kw={'xticks':(),'yticks':()})

for model , name, ax in zip(models,names,axes):
    ax.set_title(name)
    ax.plot(model[:,:3],'-')

plt.show()