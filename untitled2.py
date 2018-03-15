# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:05:44 2018

@author: gatien.tafforeau
"""

import numpy as np
import matplotlib as plt
import time
from sklearn.decomposition import PCA
from sklearn import svm

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
  
print ("Classifieur Bayésien")
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
print("Bayésien + PCA")

myPCA = PCA(n_components=50).fit(x0);
x0PCA = myPCA.transform(x0);

uPCA = []
detPCA = []
invPCA = []
varPCA = []
         
print ("Classifieur Bayésien + PCA")
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
     
print("Temps d'execution classifieur Bayésien avec PCA : " +  str(time.time() - time2));
print ("Erreur classifieur Bayésien avec PCA :" + str(float(errorsPCA)/np.size(y1)));
        
" Question 3 "         
print ("Classieur Support Vector Machines")
clf = svm.SVC()
clf.fit(x0,y0)
clf.decision_function(x1)