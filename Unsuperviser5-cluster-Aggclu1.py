# -*- coding: UTF-8 -*-
"""
file:Unsuperviser5-cluster-Aggclu1.py
data:2019-07-2516:53
author:Grey
des:
"""
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import mglearn
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,ward

X,y = make_blobs(random_state=0,n_samples=12)

linkage_array = ward(X)
dendrogram(linkage_array)
ax = plt.gca()

bounds = ax.get_xbound()
ax.plot(bounds,[7.25,7.25],"--",c="k")
ax.plot(bounds,[4,4],'--',c='k')
ax.text(bounds[1],7.25,'two clusters',va='center',fontdict={'size':15})
ax.text(bounds[1],4,'three clusters',va='center',fontdict={'size':15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
plt.show()

'''
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
mglearn.discrete_scatter(X[:,0],X[:,1],assignment)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
'''