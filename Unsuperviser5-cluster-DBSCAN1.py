# -*- coding: UTF-8 -*-
"""
file:Unsuperviser5-cluster-DBSCAN1.py
data:2019-07-269:25
author:Grey
des:
"""
from sklearn.datasets import make_blobs,make_moons
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score,silhouette_score
from sklearn.preprocessing import StandardScaler
import mglearn
import matplotlib.pyplot as plt
import numpy as np
'''
X,y = make_blobs(random_state=0,n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print("Cluster memberships:\n{}".format(clusters))
mglearn.plots.plot_dbscan()

plt.show()
'''

X,y = make_moons(n_samples=200,noise=0.05,random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fig,axes = plt.subplots(1,4,figsize=(15,3),subplot_kw={'xticks':(),'yticks':()})

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
algorithms = [KMeans(n_clusters=2),AgglomerativeClustering(n_clusters=2),DBSCAN()]
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0,high=2,size=len(X))

axes[0].scatter(X_scaled[:,0],X_scaled[:,1],c=random_clusters,cmap=mglearn.cm3,s=60)
axes[0].set_title("Random assignment - ARI:{:.2f}".format(silhouette_score(X_scaled,random_clusters)))

for ax,algorithm in zip(axes[1:],algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm3,s=60)
    ax.set_title("{}-ARI:{:.2f}".format(algorithm.__class__.__name__,silhouette_score(X_scaled,clusters)))


'''
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm2,s=60)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
'''
plt.show()
