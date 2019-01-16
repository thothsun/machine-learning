# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/1/15 11:54 PM
@file: multivariable_linear_regression.py
@desc: 多变量线性回归
"""
import numpy as np
from sklearn import datasets

dataset = datasets.load_boston()
print(dataset.get('feature_names'))
# 使用第6-8列feature，即AGE(1940年以前建成的自住单位的比例),
# DIS(距离五个波士顿就业中心的加权距离),RAD(距离高速公路的便利指数)
X = dataset.data[:, 5:8]
# 为X增加一列全为1，来求偏置项
X = np.column_stack((X, np.ones(len(X))))
y = dataset.target

# 划分训练集和测试集
X_train = X[:-20]
X_test = X[-20:]
y_train = y[:-20]
y_test = y[-20:]

X_train = np.mat(X_train)
y_train = np.mat(y_train).T

xTx = X_train.T * X_train
w = 0
if np.linalg.det(xTx) == 0.0:
    print('xTx不可逆')
else:
    w = np.ravel(xTx.I * (X_train.T * y_train))
coef_ = w[:-1]
intercept_ = w[-1]

# 去掉添加的那一列1
X_train = X_train[:, 0:3]
X_test = X_test[:, 0:3]

y_test_pred = coef_[0] * X_test[:, 0] + coef_[1] * X_test[:, 1] + coef_[2] * X_test[:, 2] + intercept_

# 矩阵转回数组
X_train = np.ravel(X_train).reshape(-1, 3)
y_train = np.ravel(y_train)

print('Coefficients: ', coef_)
print('Intercept:', intercept_)
print('the model is: y = ', coef_, '* X + ', intercept_)
# 均方误差
print("Mean squared error: %.2f" % np.average((y_test - y_test_pred) ** 2))