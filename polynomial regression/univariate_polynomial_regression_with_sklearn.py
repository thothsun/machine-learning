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
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import warnings
from sklearn import linear_model

warnings.filterwarnings(action="ignore", module="sklearn")

dataset = pd.read_csv('dataset.csv')
X = np.asarray(dataset.get('x'))
y = np.asarray(dataset.get('y'))

# 划分训练集和测试集
X_train = X[:-3]
X_test = X[-3:]
y_train = y[:-3]
y_test = y[-3:]

# 如果用线性模型拟合
regr = linear_model.LinearRegression()
regr.fit(np.asarray(X_train).reshape(-1, 1), y_train)
y_test_pred = regr.predict(np.asarray(X_test).reshape(-1, 1))
print('Coefficients: ', regr.coef_)
print('Intercept:', regr.intercept_)
print('the model is: y = ', regr.intercept_, ' + ', regr.coef_[0], '* X')
# 均方误差
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
# r2 score，0,1之间，越接近1说明模型越好，越接近0说明模型越差
print('Variance score: %.2f' % r2_score(y_test, y_test_pred), '\n')

# # fit_intercept 为 True
# model1 = Pipeline([('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression(fit_intercept=True))])
# model1 = model1.fit(X_train[:, np.newaxis], y_train)
# y_test_pred1 = model1.named_steps['linear'].intercept_ + model1.named_steps['linear'].coef_[1] * X_test
# print('Coefficients: ', model1.named_steps['linear'].coef_)
# print('Intercept:', model1.named_steps['linear'].intercept_)
# print('the model is: y = ', model1.named_steps['linear'].intercept_, ' + ', model1.named_steps['linear'].coef_[1],
#       '* X')
# # 均方误差
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred1))
# # r2 score，0,1之间，越接近1说明模型越好，越接近0说明模型越差
# print('Variance score: %.2f' % r2_score(y_test, y_test_pred1), '\n')

# fit_intercept 为 False
model1 = Pipeline([('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression(fit_intercept=False))])
model1 = model1.fit(X_train[:, np.newaxis], y_train)
y_test_pred1 = model1.named_steps['linear'].coef_[0] + model1.named_steps['linear'].coef_[1] * X_test
print('Coefficients: ', model1.named_steps['linear'].coef_)
print('Intercept:', model1.named_steps['linear'].intercept_)
print('the model is: y = ', model1.named_steps['linear'].coef_[0], '+', model1.named_steps['linear'].coef_[1], '* X')
# 均方误差
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred1))
# r2 score，0,1之间，越接近1说明模型越好，越接近0说明模型越差
print('Variance score: %.2f' % r2_score(y_test, y_test_pred1), '\n')

model2 = Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression(fit_intercept=False))])
model2 = model2.fit(X_train[:, np.newaxis], y_train)
y_test_pred2 = model2.named_steps['linear'].coef_[0] + model2.named_steps['linear'].coef_[1] * X_test + \
               model2.named_steps['linear'].coef_[2] * X_test * X_test
print('Coefficients: ', model2.named_steps['linear'].coef_)
print('Intercept:', model2.named_steps['linear'].intercept_)
print('the model is: y = ', model2.named_steps['linear'].coef_[0], '+', model2.named_steps['linear'].coef_[1], '* X + ',
      model2.named_steps['linear'].coef_[2], '* X^2')
# 均方误差
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred2))
# r2 score，0,1之间，越接近1说明模型越好，越接近0说明模型越差
print('Variance score: %.2f' % r2_score(y_test, y_test_pred2), '\n')

plt.xlabel('x')
plt.ylabel('y')
# 画训练集的散点图
plt.scatter(X_train, y_train, alpha=0.8, color='black')
# 画模型
# 线性模型拟合结果
plt.plot(X_train, regr.coef_ * X_train + regr.intercept_, color='yellow', linewidth=3)
# 多项式模型拟合结果，degree=1 时，与线性模型重合
plt.plot(X_train, model1.named_steps['linear'].coef_[0] + model1.named_steps['linear'].coef_[1] * X_train,
         color='black',
         linewidth=1)
# 多项式模型拟合结果，degree=2 时
plt.plot(X_train, model2.named_steps['linear'].coef_[0] + model2.named_steps['linear'].coef_[1] * X_train +
         model2.named_steps['linear'].coef_[2] * X_train * X_train, color='green',
         linewidth=1)
plt.show()
