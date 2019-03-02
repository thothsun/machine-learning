# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/2/12 8:57 PM
@file: univariate_polynomial_regression_with_sklearn.py
@desc: 
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import random

X = np.arange(10)
print(X)
y = 3 + 2 * X + X ** 2 + X ** 3 + random.uniform(-1,1)
print(y)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, y, alpha=0.8, color='black')
plt.show()

model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(X[:, np.newaxis], y)
print(model.named_steps['linear'].coef_)