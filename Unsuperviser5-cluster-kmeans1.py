# -*- coding: UTF-8 -*-
"""
file:Unsuperviser5-cluster-kmeans1.py
data:2019-07-259:47
author:Grey
des:
"""
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
from  sklearn.datasets import make_moons

X,y = make_moons(n_samples=200,noise=0.05,random_state=0)
kmeans = KMeans(n_clusters=10,random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred,cmap='Paired',s=60)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker="^",c=range(kmeans.n_clusters),s=60,linewidths=2,cmap='Paired')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
print("Cluster memberships:\n{}".format(y_pred))
'''
X,y = make_blobs(random_state=170,n_samples=600)
rng = np.random.RandomState(74)
transformation = rng.normal(size=(2,2))
X = np.dot(X,transformation)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred,cmap=mglearn.cm3)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='^',c=[0,1,2],s=100,linewidths=2,cmap=mglearn.cm3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
'''
'''
X_varied,y_varied = make_blobs(n_samples=200,cluster_std=[1.0,2.5,0.5],random_state=170)
y_pred = KMeans(n_clusters=3,random_state=0).fit_predict(X_varied)

mglearn.discrete_scatter(X_varied[:,0],X_varied[:,1],y_pred)
plt.legend(['cluster 0','cluster 1','cluster 2'],loc='best')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
'''

'''
X,y = make_blobs(random_state=1)


fig,axes = plt.subplots(1,2,figsize=(10,5))
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0],X[:,1],assignments,ax=axes[0])
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_
mglearn.discrete_scatter(X[:,0],X[:,1],assignments,ax=axes[1])


plt.show()
'''