# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:05:11 2021

@author: YoSke
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

X = np.load("train_x.npy")
Y = np.load("train_y.npy")
XX = np.load("test_x.npy")

#%%---------------------Preparing data
def divideX(X,x1,x2,y1,y2):
    x = list()
    lambdas = list()
    for i in range(x1,x2+1):
        for j in range(y1,y2+1):
            for z in range(0,103):
                lambdas.append(X[i][j][z])
            x.append(lambdas)
            lambdas = list()
    return x

def divideY(Y,x1,x2,y1,y2):
    y = np.zeros((x2-x1+1,y2-y1+1))
    k = 0
    for i in range(x1,x2+1):
        l = 0
        for j in range(y1,y2+1):
            y[k,l] = Y[i,j]
            l+=1
        k+=1
    return y

def extract(x,y,knbmax):
    xx = list()
    yy = list()
    knb = 0
    for k in range(0,9+1):
        for i in range(0,len(x)):
            if (knb < knbmax) and (y[i] == k):
                xx.append(x[i])
                yy.append(int(y[i]))
                knb += 1
        knb = 0
    return(xx,yy)

def extract1(x,y,knbmax0, knbmax1, knbmax2, knbmax3, knbmax4, knbmax5, knbmax6, knbmax7, knbmax8, knbmax9):
    xx = list()
    yy = list()
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax0) and (y[i] == 0):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax1) and (y[i] == 1):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax2) and (y[i] == 2):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax3) and (y[i] == 3):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax4) and (y[i] == 4):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax5) and (y[i] == 5):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax6) and (y[i] == 6):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax7) and (y[i] == 7):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax8) and (y[i] == 8):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
    knb = 0
    for i in range(0,len(x)):
        if (knb < knbmax9) and (y[i] == 9):
            xx.append(x[i])
            yy.append(int(y[i]))
            knb += 1
            
    return(xx,yy)

def derivate(x):
    xx = list()
    liste = list()
    for i in range(0,len(x)):
        for k in range(0,len(x[i])-1):
            liste.append( x[i][k+1]-x[i][k] )
        xx.append(liste)
        liste = list()
    return xx

band1 = 120       #0
band2 = 200     #309

band3 = 0        #0
band4 = 339      #339

trainortest = "test"

x_train = divideX(X,band1,band2,band3,band4)
if trainortest == "train":
    x_test = divideX(X,0,309,0,339)
if trainortest == "test":
    x_test = divideX(XX,0,299,0,339)
y = divideY(Y,band1,band2,band3,band4)
y = np.concatenate(y)

#x_train,y = extract(x_train,y,1)
#x_train,y = extract1(x_train,y,100,30,500,70,50,50,50,50,50,100)
x_train = derivate(x_train)
x_train = derivate(x_train)

x_test = derivate(x_test)
x_test = derivate(x_test)

#print(y)

print("Preparing data done")
#%%------------------------Predict
def scale(x):
    x = np.interp(x, (np.amin(x), np.amax(x)), (-1, +1))
    x1 = list()
    for k in x:
        x1.append(list(k))
    return x1

clf = svm.SVC(kernel='poly', gamma = 1)#, gamma = 0.01) # kernel = Linear, poly or rbf
#clf = GaussianNB()
clf.fit(x_train,y)
result = clf.predict(x_test)

#x_train = scale(x_train)
#x_test = scale(x_test)
#clf = MLPClassifier(hidden_layer_sizes=(10,50,50,50,10), max_iter=1000,activation = 'relu',solver='adam',random_state=0)
#clf.fit(x_train,y)
#result = clf.predict(x_test)
                            #                        IL FAUT ETRE PLUS OU MOINS
#error = result-y#                         PATIENT ICI EN FONCTION DES PARAMETRES CHOISIS
RESULT = result
if trainortest == "train":
    result = result.reshape(310,340)
if trainortest == "test":
    result = result.reshape(300,340)
print("Predict done")

#%%----------------------Display result and accuracy
plt.imshow(result, interpolation='nearest')
plt.show()

if trainortest == "train":
    end = 310
if trainortest == "test":
    end = 300

accuracy = 0
for i in range(0,end):
    for j in range(0,340):
        if Y[i,j] == result[i,j]:
            accuracy += 1
accuracy = 100*(accuracy/(end*340))
print("Accuracy : ",accuracy,"%")

#%%---------------------------Save prediction

if trainortest == "test":
    Ypredict = result
    
    Ypredict = np.concatenate(Ypredict)
    Ypredict = list(Ypredict)
    
    index = 0
    YP = np.zeros((340*300,2))
    for k in Ypredict:
        YP[index,0] = index+1
        YP[index,1] = k
        index += 1
    Ypredict = YP
    
    np.savetxt("answer.csv", Ypredict, delimiter = ',')