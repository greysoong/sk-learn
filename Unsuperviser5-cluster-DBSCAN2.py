# -*- coding: UTF-8 -*-
"""
file:Unsuperviser5-cluster-DBSCAN2.py
data:2019-07-2614:44
author:Grey
des:
"""
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import mglearn
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7)
image_shape = people.images[0].shape

mask = np.zeros(people.target.shape,dtype = np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people/255

X_train,X_test,y_train,y_test = train_test_split(X_people,y_people,stratify=y_people,random_state=0)

pca = PCA(n_components=100,whiten=True,random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)


'''
for eps in [1,3,5,7,9,11,13]:
    print("\neps={}".format(eps))
    dbscan = DBSCAN(eps=eps,min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("Clusters present:{}".format(np.unique(labels)))
    print("Cluster sizes:{}".format(np.bincount(labels+1)))
'''
'''
dbscan =DBSCAN(min_samples=3,eps=7)
labels = dbscan.fit_predict(X_pca)
'''
km = KMeans(n_clusters=10,random_state=0)
labels_km = km.fit_predict(X_pca)
print("Cluster sizes k-means:{}".format(np.bincount(labels_km)))
'''
fig,axes = plt.subplots(2,5,subplot_kw={'xticks':(),'yticks':()},figsize=(12,4))
for center,ax in zip(km.cluster_centers_,axes.ravel()):
    ax.imshow(pca.inverse_transform(center).reshape(image_shape),vmin=0,vmax=1)

plt.show()
'''

agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)
print("Cluster sizes agglomerative clustering:{}".format(np.bincount(labels_agg)))

print("ARI:{:.2f}".format(adjusted_rand_score(labels_agg,labels_km)))


'''
for cluster in range(max(labels)+1):
    mask = labels == cluster
    n_images = np.sum(mask)
    fig,axes = plt.subplots(1,n_images,figsize=(n_images*1.5,4),subplot_kw={'xticks':(),'yticks':()})
    for image,label,ax in zip(X_people[mask],y_people[mask],axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])
'''
plt.show()

'''
dbscan =DBSCAN(min_samples=3,eps=15)
labels = dbscan.fit_predict(X_pca)
print("Unique labels:{}".format(np.unique(labels)))

print("Number of points per cluster:{}".format(np.bincount(labels+1)))
noise = X_people[labels==-1]



fig,axes = plt.subplots(3,9,subplot_kw={"xticks":(),"yticks":()},figsize=(12,4))

for image,ax in zip(noise,axes.ravel()):
    ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)

plt.show()
'''