# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:34:11 2022

@author: user
"""

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import random
# Step 1: training data
X = [i for i in range(10)]
Y = [random.gauss(x,0.75) for x in X]
X = np.asarray(X)
Y = np.asarray(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]
plt.scatter(X,Y)
# Step 2: define and train a model
model = linear_model.LinearRegression()
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
# Step 3: prediction
x_new_min = 0.0
x_new_max = 10.0
X_NEW = np.linspace(x_new_min, x_new_max, 100)
X_NEW = X_NEW[:,np.newaxis]
Y_NEW = model.predict(X_NEW)
plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
plt.grid()
plt.xlim(x_new_min,x_new_max)
plt.ylim(0,20)
plt.title("Simple Linear Regression using scikit-learn and python 3",fontsize=10)
plt.xlabel('Abscise')
plt.ylabel('Ordonnee')
plt.savefig("simple_linear_regression.png", bbox_inches='tight')
plt.show()
