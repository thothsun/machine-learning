# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/1/15 9:28 PM
@file: bivariate linear regression.py
@desc: 双变量线性回归
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action="ignore", module="matplotlib")

dataset = pd.read_csv('dataset.csv')
X = dataset.get(['room_num', 'area'])
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
X_train = X_train[:, 0:2]
X_test = X_test[:, 0:2]

y_test_pred = coef_[0] * X_test[:, 0] + coef_[1] * X_test[:, 1] + intercept_

# 矩阵转回数组
X_train = np.ravel(X_train).reshape(-1, 2)
y_train = np.ravel(y_train)

print('Coefficients: ', coef_)
print('Intercept:', intercept_)
print('the model is: y = ', coef_, '* X + ', intercept_)
# 均方误差
print("Mean squared error: %.2f" % np.average((y_test - y_test_pred) ** 2))

x0, x1 = np.meshgrid(np.asarray(X_train[:, 0]), np.asarray(X_train[:, 1]))
y = np.asarray(coef_[0] * x0 + coef_[1] * x1 + intercept_)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('room_num')
ax.set_ylabel('area')
ax.set_zlabel('price')
# # 画训练集的散点图
ax.scatter(np.asarray(X_train)[:, 0], np.asarray(X_train)[:, 1], np.asarray(y_train), alpha=0.8, color='black')
# 画模型，三维空间中的一个平面
ax.plot_surface(x0, x1, y, shade=False)
plt.show()
