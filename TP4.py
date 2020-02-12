#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:41:59 2020

@author: CHEKINI Hakima
AIR
"""
import random
from sklearn import *
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# K C'est le nombre de klusters
# X c est la matrice des donnees iris X
#Q1 + Q2
def k_moyenne(X, k):
    centres  = []
    for i in range(0,k):
        centres.append(X[random.randint(0, X.shape[0]-1)])
    min_indexes =[]
    for i in range(0,300):
        dist_centres = euclidean_distances(centres, X)
    
        traspose = np.transpose(dist_centres)
        sort = np.argsort(traspose)  
        min_indexes = sort[:,0]
        centres = []
        for j in range(0, k):
            maat = sort[:,0] == j
            cc = X[maat, :]
            centres.append(np.mean(cc, axis = 0))
    
    
    return min_indexes
    
min_indexes= k_moyenne(X, 5)
plt.figure()
plt.scatter(iris.data[:,0],iris.data[:,1], c=min_indexes)
plt.title('k_moyennes')

kmeans = KMeans(n_clusters=5).fit(X)
plt.figure()
plt.scatter(iris.data[:,0],iris.data[:,1], c=kmeans.labels_)
plt.title('Kmeans de sklearn')

#Q3
def silhouette(X):
    score=0
    max_global=0
    nb_clust=0
    for i in range(2,5):
        for j in range (0,7):
            score=silhouette_score(X, k_moyenne(X, i))
            if(max_global<score):
                max_global=score
                nb_clust=i
    print ( "max global ",max_global)
    
    return nb_clust

res=silhouette(X)
 # meilleure solution: nombre de kluster = 2
print("meilleur sol est le cluster ",res)


# Q4 La différence entre PCA et LDA:
# PCA est un model d'apprentissage non supervisé alors que LDA c'est un  model d'apprentissage non supervisé
# c'est à dire les classes sont connues

target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, Y).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.title('PCA')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,
                label=target_name)

plt.title('LDA')

plt.show()
 