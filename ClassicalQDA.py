#!/usr/bin/env python
# coding: utf-8

# In[52]:


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
from sklearn.metrics import confusion_matrix, classification_report, precision_score



class QDA(object):
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
    def check_norm(dataT):
        n,p=dataT.shape
        M=[]
        for i in range(dataT.shape[0]):
            M.append(list(dataT.mean(axis=0)))  
        Sigma=dataT.cov()
        SigmaInv=inv(Sigma)
        dataM=dataT.values
        diff=dataM-M
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




    #To Check Equality of Variances
    def boxm(A,B):
        if(A.shape[1]!=B.shape[1]):
            print('Number of variables are differnt\n')
        else:
            p=A.shape[1]
        CovG=A.cov()
        CovB=B.cov()
        n1=A.shape[0]
        n2=B.shape[0]
        CovPooled=(((n1-1)/(n1+n2-2))*CovG)+(((n2-1)/(n1+n2-2))*CovB)
        print('We want to Test,\n\t H0: Variance of First Population and Second Population are Equal \n\t H1: Variances are Different\n')
        c1=((2*p*p)+(3*p)-1)/(6*(p+1))
        u=(((1/(n1-1))+(1/(n2-1)))-(1/(n1+n2-2)))*c1
        c2=((n1+n2-2)*(math.log(det(CovPooled))))-((n1-1)*(math.log(det(CovG))))+((n2-1)*(math.log(det(CovB))))
        C=(1-u)*c2
        print('BoxM Test Statistics =',C,'\n')
        chib=chi2.isf(q=0.5, df=0.05*p*(p+1))
        print('Chisq Table value =',chib,'\n')




    #To Just fit model and calculate score function
    def fit(y,X,c12=1,c21=1,plot=True,obs=0,names=False):
        #standarization
        Xmean=X.mean()
        Xstd=X.std()
        for i in range(X.shape[0]) :
            X.iloc[i,:]=(X.iloc[i,:]-Xmean)/Xstd
        
        GoodL=X[y==1]
        BadL=X[y==0]
        MeanG=GoodL.mean()
        MeanB=BadL.mean()
        CovG=GoodL.cov()
        CovB=BadL.cov()
        CovGInv=inv(CovG)
        CovBInv=inv(CovB)
        k=0.5*((math.log(det(CovG)/det(CovB)))+((np.ndarray.transpose(MeanG.values).dot(CovGInv).dot(MeanG.values))-(np.ndarray.transpose(MeanB.values).dot(CovBInv).dot(MeanB.values))))
        t1=CovGInv-CovBInv
        t2=np.ndarray.transpose(MeanG.values).dot(CovGInv)-np.ndarray.transpose(MeanB.values).dot(CovBInv)
        pd.options.display.float_format = '{:.4f}'.format
        #To calculate weight
        One=[1]*len(t2)
        One=pd.DataFrame(data=One)
        Weight=np.ndarray.transpose(One.values).dot(-0.5*t1).dot(One.values)+t2-k*np.ndarray.transpose(One.values)
        df=(pd.DataFrame(data=Weight,columns=GoodL.columns))
        display(df)
        IdealCreditScore=math.log(c12/c21)
        display('Ideal Credit Score=',IdealCreditScore)
        def CreditScore(x):
            return (-0.5*np.ndarray.transpose(x).dot(t1).dot(x))+(t2.dot(x))-k;
        scores=[]
        PredictManual=[]
        dataT=GoodL+BadL
        n=len(dataT)
        for i in range(n):
            scores.append(CreditScore(X.ix[i].values))
            if scores[i]>IdealCreditScore :
                PredictManual.append(1)
            else:
                PredictManual.append(0)
        for i in range(len(scores)):
            if(scores[i]>100):
                scores[i]=100
            elif(scores[i]<-40):
                scores[i]=-40
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
            plt.axhline(y=IdealCreditScore, color='r', linestyle='--')
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
        GoodL=X[y==1]
        BadL=X[y==0]
        MeanG=GoodL.mean()
        MeanB=BadL.mean()
        CovG=GoodL.cov()
        CovB=BadL.cov()
        CovGInv=inv(CovG)
        CovBInv=inv(CovB)
        k=0.5*((math.log(det(CovG)/det(CovB)))+((np.ndarray.transpose(MeanG.values).dot(CovGInv).dot(MeanG.values))-(np.ndarray.transpose(MeanB.values).dot(CovBInv).dot(MeanB.values))))
        t1=CovGInv-CovBInv
        t2=np.ndarray.transpose(MeanG.values).dot(CovGInv)-np.ndarray.transpose(MeanB.values).dot(CovBInv)
        return (-0.5*np.ndarray.transpose(x).dot(t1).dot(x))+(t2.dot(x))-k;

