
"""
TP1
@author: CHEKINI Hakima
AIR
"""

from sklearn import *
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

#Manipulation d un jeu de donnees
iris = datasets.load_iris()

#les donnees
data = iris.data
print (data)

#les noms des variables
print (iris.feature_names)

#le nom des classes
nomClasses = iris.target_names
print (nomClasses)

#vecteur de numero de classe
numClasses = iris.target
print (iris.target)

#afficher le nom de classe pour chaque donnees
className = iris.target_names[iris.target]
print (className)

# afficher la moyenne de chaque variable
print (iris.data.mean(0))

#afficher l ecart type de chaque variable
std = iris.data.std(0)
print (std)

#le min et le max de chaque variable
minn = iris.data.min(0)
print (minn)

maxx = iris.data.max(0)
print (maxx)

#le nombre des donnees
nbDonnes = iris.data.shape[0]
print (nbDonnes)

#le nombres de classes
nbrClasses = iris.target_names.size
print (nbrClasses)

#le nombre de variables
nbrVariables = iris.data.shape[1]
print (nbrVariables)

#Geneation de donnees et affichage

# n_samples = nombre de donnees = nombres de lignes
# n_features = nombre de variables = nombres de colonnes
#centers = nombre de groupes = targets 
don, var = make_blobs(n_samples=1000, n_features=2, centers=4)

plt.figure(figsize=(7,7))
plt.title("1000 donnees")
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(don[:,0], don[:,1], c=var)
plt.show()

lignes, colonnes = make_blobs(n_samples=100, n_features=2, centers=2)
plt.figure(figsize=(7,7))
plt.title("100 donnees")
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(lignes[:,0], lignes[:,1], c=colonnes)
plt.show()

lignes2, colonnes2 = make_blobs(n_samples=500, n_features=2, centers=3)
plt.figure(figsize=(7,7))
plt.title("500 donnees")
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(lignes2[:,0], lignes2[:,1], c=colonnes2)
plt.show()

#fusion des deux jeux de donnees
concat_array_v= np.vstack((lignes, lignes2))
concat_array_h = np.hstack((colonnes, colonnes2))

plt.figure(figsize=(7,7))
plt.title("Concatenation des jeux de donnes")
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(concat_array_v[:,0], concat_array_v[:,1], c=concat_array_h)
plt.show()