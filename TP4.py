#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:41:59 2020
@author: CHEKINI Hakima et Ben Saada Meriam
AIR3 et Info3
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
import csv
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering

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

# 4.Utilisation d une ACP"
pca = PCA()
IrisPCA = pca.fit_transform(X)
fig = plt.figure()
plt.scatter(IrisPCA[:, 0], IrisPCA[:, 1], c=k_moyenne(X, 3), s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA')

# Utilisation d une ADL"
lda = LinearDiscriminantAnalysis(n_components=4)
irisLDA = lda.fit(X, Y).transform(X)
fig = plt.figure()
plt.scatter(irisLDA[:, 0], irisLDA[:, 1], c=k_moyenne(X, 3), s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('LDA')
plt.show()

# Autre methode
print("Deuxième méthode: ")
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, Y).transform(X)
# Percentage of variance explained for each components
print('ratio de variance expliqué (deux premières composantes): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.title('PCA 2')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,
                label=target_name)

plt.title('LDA 2')

plt.show()

# B) Analyse des donnees choix projet:
def analyseDonnes():
    print("B) Analyse des donnees choix projet:")
    C = []
    M = []
    fist = True
    i=0

    with open('choixprojetstab.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
            M.append([])
            if fist == True:
                fist = False
            else:
                C.append(row[1])

                for l in range(2,len(row)):
                    M[i].append(int(row[l]))
                i=i+1
    return(C,M)
    
C,M = analyseDonnes()
print (C,"\n\n", M)
print("\n")

#c1 = AgglomerativeClustering(n_clusters=2).fit(M)
#y = c1.labels_
#print ("les classe sont \n",y)
#print ("L'indice de silhouette pour l'algorithme Agglomerative Clustering (pour "+str(unique(y).size)+" clusters) est:  \n" + str(silhouette_score(M, y)))
#

#print("\n")
#c2 = AffinityPropagation().fit(M)
#z = c2.labels_
#print ("les classe sont \n",z)
#print ("L'indice de silhouette pour l'algorithme Affinity Propagation est:\n  " + str(silhouette_score(M, z)))


#print("\n")
#c3 = SpectralClustering(n_clusters=4, assign_labels="discretize", random_state=0).fit(M)
#w = c3.labels_
#print ("les classe sont \n",w)
#print ("L'indice de silhouette pour l'algorithme Spectral Clustering (pour "+str(unique(w).size)+" clusters) est: \n " + str(silhouette_score(M, w)))
#
