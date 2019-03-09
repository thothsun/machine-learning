# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/2/12 8:56 PM
@file: polynomial_regression.py
@desc: 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')
X = np.asarray(dataset.get('x')).reshape(-1, 1)
y = np.asarray(dataset.get('y'))

# 扩展X
Z = np.column_stack((X, X ** 2))
# 为X增加一列全为1，来求偏置项
Z = np.column_stack((Z, np.ones(len(X))))

# 划分训练集和测试集
Z_train = Z[:-3]
Z_test = Z[-3:]
y_train = y[:-3]
y_test = y[-3:]

Z_train = np.mat(Z_train)
y_train = np.mat(y_train).T

xTx = Z_train.T * Z_train
w = 0
if np.linalg.det(xTx) == 0.0:
    print('xTx不可逆')
else:
    w = np.ravel(xTx.I * (Z_train.T * y_train))
coef_ = w[:-1]
intercept_ = w[-1]

# 去掉添加的那一列1
Z_train = Z_train[:, 0:2]
Z_test = Z_test[:, 0:2]

y_test_pred = coef_[0] * Z_test[:, 0] + coef_[1] * Z_test[:, 1] + intercept_

# 矩阵转回数组
Z_train = np.ravel(Z_train)
y_train = np.ravel(y_train)

print('Coefficients: ', coef_)
print('Intercept:', intercept_)
print('the model is: y = ', coef_[0], '* X + ', coef_[1], '* X^2 + ', intercept_)
# 均方误差
print("Mean squared error: %.2f" % np.average((y_test - y_test_pred) ** 2))

plt.xlabel('x')
plt.ylabel('y')
# 画训练集的散点图

X_train = X[:-3]
plt.scatter(X_train, y_train, alpha=0.8, color='black')
# 画模型
plt.plot(X_train, intercept_ + coef_[0] * X_train + coef_[1] * X_train * X_train, color='green', linewidth=1)

plt.show()
