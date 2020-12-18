# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:51:50 2019

@author: shahr
"""

import seaborn as sn
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

C1 = 0.001

CF = [];
CF1 = np.array([[93,0,0,0,0,3,3,0,1,0],\
       [ 0,98,1,1,0,0,0,0,0,0],\
       [ 0,2,89,1,0,0,1,2,5,0],\
       [ 0,4,5,82,0,3,2,0,3,1],\
       [ 0,3,0,0,97,0,0,0,0,0],\
       [ 0,3,2,0,2,90,1,0,0,2],\
       [ 2,1,8,0,1,5,83,0,0,0],\
       [ 1,1,0,0,2,0,0,87,0,9],\
       [ 1,4,0,9,0,6,0,0,79,1],\
       [ 1,4,1,2,7,2,0,0,0,83]])


C2 = 0.1
CF2 = np.array([[95,0,1,1,0,1,1,0,0,1]\
                ,[0,98,1,0,0,0,0,0,1,0]\
                ,[1,1,91,2,0,0,0,2,3,0]\
                ,[1,3,8,78,0,2,2,0,6,0]\
                ,[0,3,0,0,91,0,1,0,1,4]\
                ,[1,2,0,2,4,85,1,0,4,1]\
                ,[3,0,9,0,2,2,84,0,0,0]\
                ,[1,0,1,1,4,0,0,84,1,8]\
                ,[3,7,0,4,0,4,2,0,79,1]\
                ,[1,4,1,3,5,1,0,3,2,80]])

C3 = 100
CF3 = np.array([[95,0,1,1,0,1,1,0,0,1]\
                ,[0,98,1,0,0,0,0,0,1,0]\
                ,[1,1,91,2,0,0,0,2,3,0]\
                ,[1,3,8,78,0,2,2,0,6,0]\
                ,[0,3,0,0,91,0,1,0,1,4]\
                ,[1,2,0,2,4,85,1,0,4,1]\
                ,[3,0,9,0,2,2,84,0,0,0]\
                ,[1,0,1,1,4,0,0,84,1,8]\
                ,[3,7,0,4,0,4,2,0,79,1]\
                ,[1,4,1,3,5,1,0,3,2,80]])

C4 = 1000000.0
CF4 = np.array([[95,0,1,1,0,1,1,0,0,1]\
                ,[0,98,1,0,0,0,0,0,1,0]\
                ,[1,1,91,2,0,0,0,2,3,0]\
                ,[1,3,8,78,0,2,2,0,6,0]\
                ,[0,3,0,0,91,0,1,0,1,4]\
                ,[1,2,0,2,4,85,1,0,4,1]\
                ,[3,0,9,0,2,2,84,0,0,0]\
                ,[1,0,1,1,4,0,0,84,1,8]\
                ,[3,7,0,4,0,4,2,0,79,1]\
                ,[1,4,1,3,5,1,0,3,2,80]])

#CF5 = np.array([[84,0,0,0,16,0,0,0,0,0]\
#                ,[0,93,0,0,7,0,0,0,0,0]\
#                ,[0,1,67,0,32,0,0,0,0,0]\
#                ,[0,0,18,6,76,0,0,0,0,0]\
#                ,[0,0,0,0,100,0,0,0,0,0]\
#                ,[2,1,1,0,96,0,0,0,0,0]])

CF6 = np.array([[90,0,0,0,0,9,1,0,0,0]\
                ,[0,98,0,1,0,1,0,0,0,0]\
                ,[0,5,74,4,11,5,1,0,0,0]\
                ,[0,11,5,70,1,12,1,0,0,0]\
                ,[0,4,0,0,95,1,0,0,0,0]\
                ,[0,9,2,5,8,75,1,0,0,0]\
                ,[1,3,8,0,3,4,81,0,0,0]\
                ,[1,8,0,0,21,0,0,70,0,0]\
                ,[2,12,2,17,9,58,0,0,0,0]\
                ,[1,6,0,2,79,3,0,9,0,0]])

CF7 = np.array([[91,0,0,0,0,5,3,0,1,0]\
                ,[0,98,0,1,0,1,0,0,0,0]\
                ,[0,2,88,1,3,0,1,0,5,0]\
                ,[0,4,5,82,0,3,1,1,3,1]\
                ,[0,3,0,0,95,0,0,0,0,2]\
                ,[0,2,3,1,2,89,1,0,0,2]\
                ,[0,1,8,0,2,6,83,0,0,0]\
                ,[1,4,1,0,1,0,0,83,0,10]\
                ,[1,4,2,9,1,6,0,0,76,1]\
                ,[1,5,1,2,7,2,0,2,0,80]])

CF8 = np.array([[91,0,0,0,0,4,3,0,2,0]\
                ,[0,98,1,1,0,0,0,0,0,0]\
                ,[0,2,90,1,1,0,1,0,5,0]\
                ,[0,2,6,83,0,4,1,0,3,1]\
                ,[0,3,0,0,95,0,0,0,0,2]\
                ,[0,2,2,1,2,90,1,0,0,2]\
                ,[2,0,8,0,2,5,83,0,0,0]\
                ,[1,3,0,0,1,0,0,85,0,10]\
                ,[1,4,0,9,0,6,0,0,79,1]\
                ,[1,4,1,2,5,2,0,2,0,83]])

CF9 = np.array([[93,0,0,0,0,3,3,0,1,0]\
                ,[0,97,0,1,0,0,0,0,2,0]\
                ,[1,2,90,1,0,0,1,2,3,0]\
                ,[0,3,6,80,0,4,1,1,4,1]\
                ,[0,2,1,0,92,1,0,0,1,3]\
                ,[1,5,1,1,3,84,1,0,2,2]\
                ,[2,0,7,0,3,5,83,0,0,0]\
                ,[1,1,1,0,2,0,0,88,0,7]\
                ,[1,4,0,3,1,4,0,0,85,2]\
                ,[1,2,2,2,8,2,0,4,1,78]])

CF10 = np.array([[95,0,0,1,0,1,1,0,2,0]\
                ,[0,97,1,1,0,0,0,0,1,0]\
                ,[2,1,88,2,0,0,1,2,4,0]\
                ,[0,3,6,82,0,4,0,0,5,0]\
                ,[0,3,1,0,93,0,0,0,1,2]\
                ,[1,6,0,2,3,84,1,0,2,1]\
                ,[3,0,6,0,2,4,85,0,0,0]\
                ,[1,0,0,0,2,2,0,88,0,7]\
                ,[1,3,2,8,0,6,0,0,79,1]\
                ,[1,3,0,4,8,0,0,2,3,79]])

CF11 = np.array([[97,0,0,1,0,2,0,0,0,0]\
                ,[0,97,1,1,0,0,0,0,1,0]\
                ,[2,1,87,3,0,0,0,2,5,0]\
                ,[0,4,6,81,0,3,0,0,6,0]\
                ,[0,2,0,0,94,1,0,0,1,2]\
                ,[1,5,0,2,4,85,1,0,1,1]\
                ,[3,1,5,0,2,4,84,0,1,0]\
                ,[1,1,0,0,1,2,0,85,1,9]\
                ,[2,4,2,6,0,2,0,0,83,1]\
                ,[1,4,0,4,7,0,0,1,2,81]])

CF = []
CF.append(CF1);
CF.append(CF2);
CF.append(CF3);
CF.append(CF4);
#CF.append(CF5);
CF.append(CF6);
CF.append(CF7);
CF.append(CF8);
CF.append(CF9);
CF.append(CF10);
CF.append(CF11);

C = [C1,C2,C3,C4]
#C.append(0.0001)
C.append(0.0002)
C.append(0.0005)
C.append(0.0007)
C.append(0.002)
C.append(0.005)
C.append(0.007)

recall = np.ndarray((11,));
precis = np.ndarray((11,));

TPR = np.ndarray((10,));
FPR = np.ndarray((10,));

recall[0] = 0;
precis[0] = 1;
for i in reversed(range(10)):
    FP = CF[i].sum(axis=0) - np.diag(CF[i])  
    FN = CF[i].sum(axis=1) - np.diag(CF[i])
    TP = np.diag(CF[i])
    TN = CF[i].sum() - (FP + FN + TP)
    
    TPR[i] = np.mean(TP/(TP+FN))
    FPR[i] = np.mean(FP/(FP+TN))
    
    #print('TPR-min = ',np.min(TP/(TP+FN)))
    #print('FPR-min = ',np.min(FP/(FP+TN)))
    #print('TPR-mean = ',np.mean(TP/(TP+FN)))
    #print('FPR-mean = ',np.mean(FP/(FP+TN)))
    #print('TPR-max = ',np.max(TP/(TP+FN)))
    #print('FPR-max = ',np.max(FP/(FP+TN)))
    #print('\n')
    
    recall[i+1] = np.mean(np.diag(CF[i]) / np.sum(CF[i], axis = 1))
    precis[i+1] = np.mean(np.diag(CF[i]) / np.sum(CF[i], axis = 0))
    
    print('Precis-min = ',np.min((np.diag(CF[i]) / np.sum(CF[i], axis = 0))))
    print('Recall-min = ',np.min((np.diag(CF[i]) / np.sum(CF[i], axis = 1))))
    print('Precis-mean = ',np.mean((np.diag(CF[i]) / np.sum(CF[i], axis = 0))))
    print('Recall-mean = ',np.mean((np.diag(CF[i]) / np.sum(CF[i], axis = 1))))
    print('Precis-max = ',np.max((np.diag(CF[i]) / np.sum(CF[i], axis = 0))))
    print('Recall-max = ',np.max((np.diag(CF[i]) / np.sum(CF[i], axis = 1))))
    
    print('\n\n\n')
    






SZ = 18;












    
#plt.plot(recall[1:],precis[1:],'o')
plt.figure(figsize = (10*3.5,7))
plt.grid(which='major',axis='both')


plt.subplot(131)
plt.plot(recall,precis,'-o')
#'(a) Precision-Recall Plot for C='+str(C);
plotTitle = '(a) Precision Recall Plot for Different Values of C'; plt.title(plotTitle,fontsize=SZ)
plt.xlabel('Recall',fontsize=SZ-2)
plt.ylabel('Precision',fontsize=SZ-2)
plt.grid(which='major')


df_cm = pd.DataFrame(CF1, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])

plt.subplot(132)
sn.heatmap(df_cm, annot=True,cmap="GnBu")
plt.title('(b) Confusion Matrix Heat-Map for S3VM and C=0.0007',fontsize=SZ)
plt.xlabel('Actual Digit',fontsize=SZ-2)
plt.ylabel('Predicted Digit',fontsize=SZ-2)
#plt.grid(which='major')

plt.subplot(133)

ratio = [300, 600, 900, 1200, 1500]
S3VM = [0.76, 0.88, 0.88, 0.89, 0.90]
SVM =  [0.76, 0.85, 0.86, 0.88, 0.88]
plt.plot(ratio,SVM,'-o')
plt.plot(ratio,S3VM,'-o')
plt.legend(['Classical SVM','Semi-Supervised SVM'])
plt.title('(c) Accruacy-Performance Measure for SVM and S3VM (C=0.0007)',fontsize=SZ)
plt.xlabel('Labeled Data Size',fontsize=SZ-2)
plt.ylabel('Accuracy',fontsize=SZ-2)
plt.grid(which='major')



























#plt.subplot(121)
#plt.plot(recall,precis,'-o')
#'(a) Precision-Recall Plot for C='+str(C);
#plotTitle = '(a) Precision Recall Plot for Different Values of C'; plt.title(plotTitle,fontsize=SZ)
#plt.xlabel('Recall',fontsize=SZ-2)
#plt.ylabel('Precision', fontsize=SZ-2)
#plt.grid(which='major')


#df_cm = pd.DataFrame(CF1, index = [i for i in "0123456789"],
#                  columns = [i for i in "0123456789"])

#plt.subplot(122)
#sn.heatmap(df_cm, annot=True,cmap="GnBu")
#plt.title('(b) Confusion Matrix Heat-Map for S3VM and C=0.001',fontsize=SZ)
#plt.xlabel('Actual Digit',fontsize=SZ-2)
#plt.ylabel('Predicted Digit',fontsize=SZ-2)
#plt.grid(which='major')

