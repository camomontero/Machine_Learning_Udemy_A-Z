#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:33:57 2018

@author: camilo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1,1) # I needed to reshape this because an error from the compiler.
sc_y = StandardScaler()
y = sc_y.fit_transform(y).reshape(-1,1)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # By default kernel='rbf'
regressor.fit(X, y)


# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) # Here we need to transform
#the 6.5 value. We do it with the transform method. But this method receives an array
# as argument, therefore we add the squarebrackets. If we add one [] it represents
# a vector. If we add [[]] it represents an array(matrix of 1 colomn 1 row).
y_pred = sc_y.inverse_transform(y_pred) # Here we need to inverse the y_pred, so that
# we see the output value in the real scale.

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()