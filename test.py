# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:52:30 2018

@author: gatien.tafforeau
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing


#Question 3
x0 = np.load('data/trn_img.npy')
y0 = np.load('data/trn_lbl.npy')

x1 = np.load('data/dev_img.npy')
y1 = np.load('data/dev_lbl.npy')


time = time.time()
print("Support Vector Machine Classifieur")

x0Scale = preprocessing.scale(x0)
clf = svm.SVC()
clf.fit(x0Scale, y0)

print("Calculation in progress")
x1Scale = preprocessing.scale(x1)
results = clf.predict(x1Scale)

taux = 0
for i, result in enumerate(results):
    if result == y1[i]:
        taux += 1 

print("Taux réussite : " + str(float(taux)/np.size(y1)))
print("Temps : " + str(time.time() - time))



#K nprint("** Neighbours **")
    
time = time.time()
print("Classifieur Neighbours")

neighbour = KNeighborsClassifier(n_neighbors = 3)
neighbour.fit(x0, y0)

print("Calculation in progress")
results = neighbour.predict(x1)

taux = 0
for i, result in enumerate(results):
    if result == y1[i]:
        taux += 1 

print("Taux réussite : " + str(float(taux)/np.size(y1)))
print("Temps : " + str(time.time() - time))


print("Classifieur Neighbours with PCA")
    
time = time.time()

myPca = PCA(n_components = 50).fit(x0)
X0pca = myPca.transform(x0)

neighbour = KNeighborsClassifier(n_neighbors = 3)
neighbour.fit(X0pca, y0)

print("Calculation in progress")
X1pca = myPca.transform(x1)
results = neighbour.predict(X1pca)

taux = 0
for i, result in enumerate(results):
    if result == y1[i]:
        taux += 1 

print("Taux réussite : " + str(float(taux)/np.size(y1)))
print("Temps : " + str(time.time() - time))



print("Classifieur Vector Machines with PCA")
    
timeTmp = time.time()

myPca = PCA(n_components = 50).fit(x0)
X0pca = myPca.transform(x0)

X0scale = preprocessing.scale(X0pca)
clf = svm.SVC()
clf.fit(X0scale, y0)

print("Calculation in progress")
x1pca = myPca.transform(x1)
x1Scale = preprocessing.scale(x1pca)
results = clf.predict(x1Scale)

taux = 0
for i, result in enumerate(results):
    if result == y1[i]:
        taux += 1 

print("Taux réussite : " + str(float(taux)/np.size(y1)))
print("Temps : " + str(time.time() - time))