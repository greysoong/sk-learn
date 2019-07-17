# -*- coding: UTF-8 -*-
"""
file:KNN1.py.py
data:2019-07-158:39
author:Grey
des:
"""
import mglearn
import matplotlib.pyplot as plt
import graphviz

#mglearn.plots.plot_knn_classification(n_neighbors=3)
#mglearn.plots.plot_knn_regression(n_neighbors=3)
#mglearn.plots.plot_linear_regression_wave()
#mglearn.plots.plot_ridge_n_samples()
#mglearn.plots.plot_linear_svc_regularization()
#mglearn.plots.plot_animal_tree()
#plt.show()
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)