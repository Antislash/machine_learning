# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:52:30 2018

@author: gatien.tafforeau
"""

import numpy as np
import matplotlib as plt
import time
from sklearn.decomposition import PCA
from sklearn import svm


x0 = np.load('data/trn_img.npy')
y0 = np.load('data/trn_lbl.npy')

x1 = np.load('data/dev_img.npy')
y1 = np.load('data/dev_lbl.npy')

" Question 3 "         
print ("Classieur Supprort Vector Machines")
clf = svm.SVC()
clf.fit(x0,y0)
dec = clf.decision_function(x1)

print ("fini")
print (dec)