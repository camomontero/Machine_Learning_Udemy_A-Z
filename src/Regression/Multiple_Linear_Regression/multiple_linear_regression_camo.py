#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:48:33 2018

@author: camilo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('50_Startups.csv')

# Matrix of features (columns of independent variables)
X=dataset.iloc[:, :-1].values # [lines rows, columns] [:means all the lines, :-1means all columns except the last one]

# Dependent variable vector
y=dataset.iloc[:, 4].values # [:, 4] [:all the lines, 4 the column index 4 which is the last column (Profit)]

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() 
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # Label each feature of the column with a value (encode each feature)

onehotencoder = OneHotEncoder(categorical_features = [3]) 
X = onehotencoder.fit_transform(X).toarray() # Takes the encoded features and create the Dummy Variables with 0s and 1s

# Avoiding the Dummy Varaibles Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (it is done if one of te ind. vble is much bigger than the others, that the others are seen as i significant)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) 

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

#Append to the matrix of features X a column of ones, which is the Xo. Formula: y=boXo+b1X1+...+bnXn
#It is needed, since the library statsmodels does not take that into consideration.
#Before, we did not need to do it since sklearn library takes care about it.
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)

X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()   # OLS = Ordinary Least Squares
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()   # OLS = Ordinary Least Squares
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()   # OLS = Ordinary Least Squares
regressor_OLS.summary()

X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()   # OLS = Ordinary Least Squares
regressor_OLS.summary()

X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()   # OLS = Ordinary Least Squares
regressor_OLS.summary()