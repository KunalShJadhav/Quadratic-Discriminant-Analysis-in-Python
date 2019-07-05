#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''This method perform Quadratic Discriminant Analysis By Classical Method'''
from __future__ import division
import sys
import math
import copy
from time import time
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2 
from sklearn.metrics import confusion_matrix



class qda(object):
    def __init__(self, x, y):
        self.funcs_ = funcs_
        #verify the dimensions
        if self.funcs_.verify_dimensions(x):
            if len(x) == len(y):
                self.len_all = len(x)
                self.dmean_ = self.funcs_.mean_nm(x, axis=0)
                self.std_ = self.funcs_.std_nm(x, axis=0)
                self.x = x
                self.y = y
                self.separate_sets()
            else:
                sys.exit()
        else:
            print('data dimensions are inaccurate..exiting..')
            sys.exit()
            
    #To Check Normality
    def check_norm(X,names=False):
        n,p=X.shape
        M=[]
        for i in range(X.shape[0]):
            M.append(list(X.mean(axis=0)))  
        Sigma=X.cov()
        SigmaInv=inv(Sigma)
        diff=(X.values)-M
        GD=[]
        for i in range(len(diff)):
            GD.append(diff[i].dot(SigmaInv).dot(diff[i]))

        chi=chi2.isf(q=0.5, df=p)
        print('ChiSquare table value is',chi,'\n')

        sum=0
        for i in range(n) :
            if GD[i] <= chi :
                 sum=sum+1

        print('Number of Observation below QQ line',sum,'\n')
        
        #Plotting
        obs=len(X)
        plt.scatter(range(obs),GD)
        plt.axhline(y=chi, color='r', linestyle='--')
        if(names==True):
            for i in range(obs):
                plt.annotate(i,(i,GD[i]))
            plt.show()
        print('\n***Interpretation : Data is considered to be Normally distributed if Half of the Observations are below/above QQ line\n\tWhile More Observations below/above QQ line denotes deviation from Normality')






    #To Just fit model and calculate score function
    def fit(y,X,c12=1,c21=1,plot=True,obs=0,names=False):
        #standarization
        Xmean=X.mean()
        Xstd=X.std()
        for i in range(X.shape[0]) :
            X.iloc[i,:]=(X.iloc[i,:]-Xmean)/Xstd
        
        A=X[y==1]
        B=X[y==0]
        MeanA=A.mean()
        MeanB=B.mean()
        CovA=A.cov()
        CovB=B.cov()
        CovAInv=inv(CovA)
        CovBInv=inv(CovB)
        k=0.5*((math.log(det(CovA)/det(CovB)))+((np.ndarray.transpose(MeanA.values).dot(CovAInv).dot(MeanA.values))-(np.ndarray.transpose(MeanB.values).dot(CovBInv).dot(MeanB.values))))
        t1=CovAInv-CovBInv
        t2=np.ndarray.transpose(MeanA.values).dot(CovAInv)-np.ndarray.transpose(MeanB.values).dot(CovBInv)
        pd.options.display.float_format = '{:.4f}'.format
        #To calculate weight
        One=[1]*len(t2)
        One=pd.DataFrame(data=One)
        Weight=np.ndarray.transpose(One.values).dot(-0.5*t1).dot(One.values)+t2-k*np.ndarray.transpose(One.values)
        df=(pd.DataFrame(data=Weight,columns=A.columns))
        display(df)
        IdealScore=math.log(c12/c21)
        display('Ideal Score=',IdealScore)
        def RealScore(x):
            return (-0.5*np.ndarray.transpose(x).dot(t1).dot(x))+(t2.dot(x))-k;
        scores=[]
        PredictManual=[]
        dataT=A+B
        n=len(dataT)
        for i in range(n):
            scores.append(RealScore(X.ix[i].values))
            if scores[i]>IdealScore :
                PredictManual.append(1)
            else:
                PredictManual.append(0)
        for i in range(len(scores)):
            if(scores[i]>IdealScore+100):
                scores[i]=IdealScore+100
            elif(scores[i]<IdealScore-100):
                scores[i]=IdealScore-100
        CM=(confusion_matrix(y,PredictManual))
        Accuracy=(CM[0,0]+CM[1,1])/len(X)
        print('Accuracy=',Accuracy)
        display('Confusion Matrix',CM)
        if(plot==True):
            if(obs==0):
                obs=len(X)
            label_color_dict = {label:idx for idx,label in enumerate(np.unique(y))}
            cvec = [label_color_dict[label] for label in y]
            plt.scatter(range(obs),scores,c=cvec)
            plt.axhline(y=IdealScore, color='r', linestyle='--')
            if(names==True):
                for i in range(obs):
                    plt.annotate(i,(i,scores[i]))
            plt.show()

    #To Predict    
    def predict(x,y,X,c12=1,c21=1):
        #standarization
        Xmean=X.mean()
        Xstd=X.std()
        for i in range(X.shape[0]) :
            X.iloc[i,:]=(X.iloc[i,:]-Xmean)/Xstd
        x=(x-Xmean)/Xstd
        x=x.values
        A=X[y==1]
        B=X[y==0]
        MeanA=A.mean()
        MeanB=B.mean()
        CovA=A.cov()
        CovB=B.cov()
        CovAInv=inv(CovA)
        CovBInv=inv(CovB)
        k=0.5*((math.log(det(CovA)/det(CovB)))+((np.ndarray.transpose(MeanA.values).dot(CovAInv).dot(MeanA.values))-(np.ndarray.transpose(MeanB.values).dot(CovBInv).dot(MeanB.values))))
        t1=CovAInv-CovBInv
        t2=np.ndarray.transpose(MeanA.values).dot(CovAInv)-np.ndarray.transpose(MeanB.values).dot(CovBInv)
        return (-0.5*np.ndarray.transpose(x).dot(t1).dot(x))+(t2.dot(x))-k;

