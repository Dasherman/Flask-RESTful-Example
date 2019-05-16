#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import scipy as sp

from sklearn.neighbors import KNeighborsRegressor
import sklearn.model_selection as ms

import pickle

np.random.seed(0)


# k-nearest neighbour training

# In[18]:


# In[74]:


class kNN(object):
    
    def __init__(self, weights='uniform'):
        self.weights = weights #mode for the weights
    
    def standardize_X(self, X, t_X=None):
        #standardizes X variables either using mean and std used for fitting or t_X (if provided).
        if np.all(t_X==None):
            mean = self.X_mean
            std = self.X_std
        else:
            mean = np.mean(t_X, axis=0)
            std = np.std(t_X, axis=0, ddof=1)
        
        return (X-mean)/std
    
    def standardize_y(self, y, t_y=None):
        #standardizes y variables either using mean and std used for fitting or t_y (if provided).
        if np.all(t_y==None):
            mean = self.y_mean
            std = self.y_std
        else:
            mean = np.mean(t_y)
            std = np.std(t_y, ddof=1)
        
        return (y-mean)/std
    
    def destandardize_X(self, X, t_X=None):
        if np.all(t_y==None):
            mean = self.y_mean
            std = self.y_std
        else:
            mean = np.mean(t_y)
            std = np.std(t_y, ddof=1)
        
        return X*std + mean
    
    def destandardize_y(self, y, t_y=None):
        if np.all(t_y==None):
            mean = self.y_mean
            std = self.y_std
        else:
            mean = np.mean(t_y)
            std = np.std(t_y, ddof=1)
        
        return y*std + mean
    
    def pred(self, X, kNN=None):
        if kNN==None:
            kNN = self.kNN
        
        y_pred = kNN.predict(self.standardize_X(X))
        y_pred = self.destandardize_y(y_pred)
        return y_pred
    
    def fit(self, X, y, m=5, min_k=1, max_k=None):
        #fits kNN. Uses m-fold CV to determine k (maximizing R^2).        
        if not max_k:
            max_k = int(np.sqrt(len(y)*len(X[0])))
        
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_std = np.std(y, axis=0)
        
        X_standardized = self.standardize_X(X)
        y_standardized = self.standardize_y(y)
        
        splitter = ms.KFold(n_splits=m)
        folds = [(train, test) for train, test in splitter.split(X)] #indices
        
        def MSE(k):
            kNN = KNeighborsRegressor(n_neighbors=k, weights=self.weights)
            MSEs = []
            for train, test in folds:
                kNN.fit(X_standardized[train], y_standardized[train])
                y_pred = self.pred(X[test], kNN=kNN)
                error = y[test] - y_pred
                MSEs.append(np.mean(error**2))
            
            return np.mean(MSEs)
        
        MSEs = list(map(MSE, range(min_k, max_k+1)))
        min_MSE = min(MSEs)
        k = MSEs.index(min_MSE)+1
        
        self.kNN = KNeighborsRegressor(n_neighbors=k, weights=self.weights)
        self.kNN.fit(X_standardized, y_standardized)
        
        return kNN
    
    def pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    


# Inlezen verzekeringsdata en dummy's maken zodat het een regressormatrix wordt.

# In[75]:


data = pd.read_csv('insurance.csv')


# In[76]:


data = pd.get_dummies(data, ['sex', 'smoker', 'region'], drop_first=True)
data.head()


# In[77]:


y = data['charges']
X = data.drop(columns='charges')
headers = X.columns


# In[78]:


y = np.array(y)
X = np.array(X)


# Nu trainen we het model.

# In[80]:


out = 'model.pickled'

model = kNN(weights='uniform')
model.fit(X, y, m=10)
model.kNN.n_neighbors


# In[83]:


model.pickle(out)


# In[ ]:




