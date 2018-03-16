# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:05:44 2018

@author: gatien.tafforeau
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import preprocessing

taille = 10

x0 = np.load('data/trn_img.npy')
y0 = np.load('data/trn_lbl.npy')

x1 = np.load('data/dev_img.npy')
y1 = np.load('data/dev_lbl.npy')

" Question 1 "
print("Classifieur Bayesien")
time1 = time.time()
u = []
#cov = []
det = []
inv = []
var = []
for i in range (0,taille):
    var = x0[y0==i]
    u.append(var.mean(axis=0))
    cov = (np.cov(var,rowvar = 0))
    det.append(np.linalg.slogdet(cov)[1])
    inv.append(np.linalg.inv(cov))

classes = [] 
  
print ("Classifieur Calculation")
errors = 0
for i,img in enumerate(x1):
     tmp = []
     for j in range(0, taille):
         tmp.append(-det[j] - np.dot(np.dot(np.transpose(img - u[j]), inv[j]), img - u[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errors += 1

print("Taux d'erreur bayésien : " + str(float(errors)/np.size(y1)))

print("Temps bayésien : " + str(time.time() - time1))



" Question 2 "
errorMat = []
components = []
print("Classifieur Bayésien + PCA with 25 component")

myPCA = PCA(n_components=25).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(25)

print("Temps d'execution classifieur Bayésien avec PCA 25 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 25 composants:" + str(float(errorsPCA)/np.size(y1)));

print("Classifieur Bayésien + PCA with 35 component")

myPCA = PCA(n_components=35).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(35)

print("Temps d'execution classifieur Bayésien avec PCA 35 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 35 composants:" + str(float(errorsPCA)/np.size(y1)));
         
print("Classifieur Bayésien + PCA with 50 component")

myPCA = PCA(n_components=50).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1
errorMat.append(float(errorsPCA)/np.size(y1))
components.append(50)
 
print("Temps d'execution classifieur Bayésien avec PCA  50 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 50 composants :" + str(float(errorsPCA)/np.size(y1)));

print("Classifieur Bayésien + PCA with 75 component")

myPCA = PCA(n_components=75).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(75)

print("Temps d'execution classifieur Bayésien avec PCA 75 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 75 composants:" + str(float(errorsPCA)/np.size(y1)));


print("Classifieur Bayésien + PCA with 100 component")

myPCA = PCA(n_components=100).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(100)

print("Temps d'execution classifieur Bayésien avec PCA 100 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 100 composants:" + str(float(errorsPCA)/np.size(y1)));

print("Classifieur Bayésien + PCA with 120 component")

myPCA = PCA(n_components=120).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(120)

print("Temps d'execution classifieur Bayésien avec PCA 120 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 120 composants:" + str(float(errorsPCA)/np.size(y1)));

print("Classifieur Bayésien + PCA with 140 component")

myPCA = PCA(n_components=140).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur calculation")
for i in range (0,taille):
    varPCA = x0PCA[y0==i]
    uPCA.append(varPCA.mean(axis=0))
    covPCA = (np.cov(varPCA,rowvar = 0))
    detPCA.append(np.linalg.slogdet(covPCA)[1])
    invPCA.append(np.linalg.inv(covPCA))



X1Pca = myPCA.transform(x1);

resultsPca = [];            

time2=time.time();

errorsPCA = 0
for i,img in enumerate(X1Pca):
     tmp = []
     for j in range(0, taille):
         tmp.append(-detPCA[j] - np.dot(np.dot(np.transpose(img - uPCA[j]), invPCA[j]), img - uPCA[j]))  
     label=np.argmax(np.array(tmp))
     if label != y1[i]:
         errorsPCA += 1

errorMat.append(float(errorsPCA)/np.size(y1))
components.append(140)

print("Temps d'execution classifieur Bayésien avec PCA 140 composants : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA 140 composants:" + str(float(errorsPCA)/np.size(y1)));

plt.plot(components, errorMat);



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


'Question 3'


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