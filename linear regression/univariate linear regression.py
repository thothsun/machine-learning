# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/1/15 7:45 PM
@file: univariate linear regression.py
@desc:
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('dataset.csv')
X = np.asarray(dataset.get('area')).reshape(-1, 1)
# 为X增加一列全为1，来求偏置项
X = np.column_stack((X, np.ones(len(X))))
y = dataset.get('price')

# 划分训练集和测试集
X_train = X[:-3]
X_test = X[-3:]
y_train = y[:-3]
y_test = y[-3:]

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
X_train = X_train[:, 0]
X_test = X_test[:, 0]

y_test_pred = coef_[0] * X_test + intercept_

# 矩阵转回数组
X_train = np.ravel(X_train)
y_train = np.ravel(y_train)

print('Coefficients: ', coef_)
print('Intercept:', intercept_)
print('the model is: y = ', coef_, '* X + ', intercept_)
# 均方误差
print("Mean squared error: %.2f" % np.average((y_test - y_test_pred) ** 2))

plt.xlabel('area')
plt.ylabel('price')
# 画训练集的散点图
plt.scatter(X_train, y_train, alpha=0.8, color='black')
# 画模型，二维空间中的一条直线
plt.plot(X_train, coef_ * X_train + intercept_, color='red', linewidth=1)
plt.show()
