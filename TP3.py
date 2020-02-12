#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP3
@author: CHEKINI Hakima
AIR
"""
from sklearn import *
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Plus Proche Voisin
def nb_occurences(x, l):
    cpt = 0
    for i in l:
        if i == x:
            cpt += 1
    return cpt

def majorite_class(cc):

    maxxx = 0   
    class_maj = 0
    for cc_i in cc:
        nb_occ = nb_occurences(cc_i, cc)
        if(nb_occ > maxxx):
            maxxx = nb_occ
            class_maj = cc_i
    
    
    return class_maj
        
    
def PPV(X,Y,k):
    #calculer la distance euclidienne de toutes les donnees
    euclid_dist = euclidean_distances(X, X)
    sort = np.argsort(euclid_dist)
    # Y[sort[:,1:3]] ici on affiche la prediction de la classe de chaque donnee par rapport a ses 2 plus 
    # proches voisins, on ecrit 3 pour 2 plus proche voisins, le 3 exclu
    pred= Y[sort[:,1: k+1]]
    #ici, on regarde pour chaque classe predite pour une donnee. si elle a deux classes, on prends la distance la plus petite
    for cc in pred:
        maj_class = majorite_class(cc)
        # La classe majoritaire se trouve maintenant dans cc[0]
        cc[0] = maj_class
            
    return pred[:,0]

    
prediction_manual = PPV(X, Y, 4)

# En testant avec k = 1, les resultats ne sont pas differents, par contre avec un k = 3 ou k = 4 on voit bien une difference
neigh = KNeighborsClassifier(n_neighbors=3)
fit = neigh.fit(X, Y)
prediction_KNeighborsClassifier = neigh.predict(X)

def PPV_erreur_prediction(X,Y, k):
    #calculer la distance euclidienne de toutes les donnees
    euclid_dist = euclidean_distances(X, X)
    sort = np.argsort(euclid_dist)
    # Y[sort[:,1:3]] ici on affiche la prediction de la classe de chaque donnee par rapport a ses 2 plus 
    # proches voisins, on ecrit 3 pour 2 plus proche voisins, le 3 exclu
    pred= Y[sort[:,1:k+1]]
    #ici, on regarde pour chaque classe predite pour une donnee. si elle a deux classes, on prends la distance la plus petite
    for cc in pred:
        maj_class = majorite_class(cc)

        # La classe majoritaire se trouve maintenant dans cc[0]
        cc[0] = maj_class
    
    j = 0
    cpt_erreur = 0
    # parcourir les classes de nos donnees Iris (iris.target) et regarder la difference avec les classes predites par notre algo
    for i in Y:
        if(i != pred[j, 0]):    
            cpt_erreur += 1
        j += 1
    # retourner le pourcentage d erreur
    return cpt_erreur * 100 / np.size(Y) * 1.1

pourcentage_erreur = PPV_erreur_prediction(X,Y,3)
print("Erreur de prediction: plus proches voisins ",pourcentage_erreur,"%")

# Bonus ==> regardez en haut

# Classifieur Bayesien Naïf

def Probclass(Y, y):  
    classes = np.unique(Y)
    longueur = len(classes)
    hauteur =  Y.shape
    Pwk = []
   
    for i in range(0, longueur):
        if classes[i] == y:
            p = float(len(Y[Y==classes[i]])-1) / float(hauteur[0]-1)
        else:
            p = float(len(Y[Y==classes[i]])) / float(hauteur[0]-1)
        Pwk.append(p)
    Pwk = np.asarray(Pwk)
    return Pwk


def ProbXiSachantW(Y,X,k, v, c):
    classe = c
    val = v
    tab= X[Y==classe, k]
    p = float(len(tab[tab==val])) / float(len(Y[Y==classe]))
     
    return p


def ProduitProb(X,Y, Pwk, i):
    P=[[],[]]
    tab = X[i,:]
    tablong = len(tab)
    classes = np.unique(Y)
    p=1
    for c in range (0, len(classes)):
        Pclas = Pwk[c]
        p=1
        for j in range(0, tablong):
            val = tab[j];
            c= classes[c]
            k=j
            px = ProbXiSachantW(Y,X,k, val, c)
            p = p*px*Pclas
        P[0].append(p)
        P[1].append(c)
    P = np.asarray(P)
    max = np.argmax(P[0,:])
    return int( P[1,max])
       
   

def CBN (X, Y):
   
    hauteur =  Y.shape
    etiquettes = []
   
    for k in range (0,hauteur[0]):
        Pwk = Probclass(Y, Y[k])
        etiquettes.append(ProduitProb(X,Y, Pwk,k ))
    return np.asarray(etiquettes)



def CBN_Err (X, Y):
   
    hauteur =  Y.shape
    etiquettes = []
   
    for k in range (0,hauteur[0]):
        Pwk = Probclass(Y, Y[k])
        etiquettes.append(ProduitProb(X,Y, Pwk,k ))
       
    Err = abs(Y - np.asarray(etiquettes))
    return  (float(np.sum(Err))/float(len(Err)))*100

labelPredit = CBN(X,Y)

pred_err = CBN_Err(X, Y)
print ("Erreur de prediction: Classifieur Bayesien ",pred_err,"%")

