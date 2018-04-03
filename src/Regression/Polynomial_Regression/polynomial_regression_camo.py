#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 21:04:44 2018

@author: camilo
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')

# Matrix of features (columns of independent variables)
X=dataset.iloc[:, 1:2].values # [lines rows, columns] [:means all the lines, :-1means all columns except the last one]
# The POSITION column is not considered in the matrix of features, since the LEVEL column already represent it, LEVEL column 
# is kind of the encoded version of the POSITION column.
# .iloc[:, 1:2] -> 1:2 takes only the column with index 1. It was written like this instead of only 1, because in this
#way it keeps being a matrix, otherwise it becomes a vector. For the independent variable we always should have a matrix.

# Dependent variable vector
y=dataset.iloc[:, 2].values # [:, 2] [:all the lines, 2 the column index 3 which is the last column (Purchased)]


# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)"""
# There is no need to split the dataset into training and test, because we have very litle observations (only 10 rows).
# ALso in order to make a very accurate prediction, we need all the dataset.

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
# No need of feature scaling since the library for the Linear Regression considers that already.

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) # We change the degree to check with which exponent the model approach better to the 
# Poly_reg object is a transformer tool that transform our matrix of features X
# into a new matrix of features that we call X_poly
# actual data
X_poly = poly_reg.fit_transform(X)
# X_poly is then a matrix of 3 columns, the column index 1 is the actual X, the column index 2 is the X^2 (polynomial)
# and the column index 0 is the constant bo added by the library. y = bo*x0 + b1*x1 + b2*x1^2

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualization the Linear Regression results

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

# Visualization the Polynomial Regression results

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
#plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
# We take the poly_reg.fit_transform(X_grid) instead of X_poly because X_poly was
# defined for the specific set of matrix of features X. In order to make this predictor
# general for every set of data we want, we use the poly_reg.fit_transform(X_grid).
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))