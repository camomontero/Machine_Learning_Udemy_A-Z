#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:14:33 2018

@author: camilo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backwardElimination(x, sl):
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog=y, exog=x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
        print regressor_OLS.summary()
    return x


data = pd.read_csv('50_Startups.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

#Encode the categorical variable (State)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelenconder_X = LabelEncoder()
X[:, 3] = labelenconder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]
sl=0.05
X_Modeled = backwardElimination(X_opt, sl)
print X_Modeled