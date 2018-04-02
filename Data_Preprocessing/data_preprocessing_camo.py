#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:03:38 2018

@author: camilo
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Data.csv')

# Matrix of features (columns of independent variables)
X=dataset.iloc[:, :-1].values # [lines rows, columns] [:means all the lines, :-1means all columns except the last one]
type(X)
# Dependent variable vector
Y=dataset.iloc[:, 3].values # [:, 3] [:all the lines, 3 the column index 3 which is the last column (Purchased)]

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer (missing_values="NaN", strategy="most_frequent", axis=0)
imputer = imputer.fit (X[:, 1:3]) # I
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data (Country and Purchased)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder=OneHotEncoder (categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)





