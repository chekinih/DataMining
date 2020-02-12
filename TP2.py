#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:41:45 2019
TP2
@author: CHEKINI Hakima
AIR
"""

 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import * 
from sklearn import preprocessing 
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#A. Normalisation de données

#creation de la matrice X
x= np.array( [[1, -1, 2],[2, 0, 0],[0, 1, -1]])
print(x)
moyenne = np.mean(x)
print(moyenne)
# moyenne = 0.44444444444444442

variance = np.var(x)
print(variance)
# variance =1.1358024691358024

mat_normalisee = preprocessing.scale(x)
print (mat_normalisee)
#mat_normalisee =   [[ 0.         -1.22474487  1.33630621]
#                   [ 1.22474487  0.         -0.26726124]
#                   [-1.22474487  1.22474487 -1.06904497]]

# les valeurs de la matrice normalisee sont proche de 0

moyenne_normalisee = np.mean(mat_normalisee)
print(moyenne_normalisee)
# moyenne_normalisee = 4.93432455389e-17
#la moyenne de la matrice normalisee est 0...

variance_normalisee = np.var(mat_normalisee)
print(variance_normalisee)
#vaaiance_normalisee = 1.0
#la variance de la matrice normalisee est 1


# B. Normalisation MinMax

x2 = np.array( [[1, -1, 2],[2, 0, 0],[0, 1, -1]])
print(x2)

#calculer la moyenne sur les variables
print(x2.mean(axis=1))

#normaliser les donnes dans l intervalle [0 1]
x2_normalisee = preprocessing.minmax_scale(x2, feature_range=(0,1))
print(x2_normalisee)

# calculer la moyenne sur les variables
print(x2_normalisee.mean(axis=1))
#On constate que la moyenne sur chaque variable est dans l'intervalle [0,1]


#C. visualisation de donnees :
    
# charger les donnees d iris
#il y a 6 possibilites (4-1)! = 6
iris = datasets.load_iris()
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.target)
plt.figure()
plt.scatter(iris.data[:,0], iris.data[:,2],  c=iris.target)
plt.figure()
plt.scatter(iris.data[:,0], iris.data[:,3],  c=iris.target)
plt.figure()
plt.scatter(iris.data[:,1], iris.data[:,2], c=iris.target)
plt.figure()
plt.scatter(iris.data[:,1], iris.data[:,3], c=iris.target )
plt.figure()
plt.scatter(iris.data[:,2], iris.data[:,3], c=iris.target)
plt.figure()
# le mieux c est quand les classes sont bien separees et distingues  et 
#ou l'intervalle des axes est tres larges

# . Reduction de dimension et visualisation de donnees

X = iris.data
Y = iris.target

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