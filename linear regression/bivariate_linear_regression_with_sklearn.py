# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/1/14 7:11 PM
@file: bivariate_linear_regression_with_sklearn.py
@desc: 双变量线性回归-使用sklearn
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 不想看到warning，添加以下代码忽略它们
warnings.filterwarnings(action="ignore", module="sklearn")
warnings.filterwarnings(action="ignore", module="matplotlib")

dataset = pd.read_csv('dataset.csv')
X = dataset.get(['room_num', 'area'])
y = dataset.get('price')

# 划分训练集和测试集
X_train = X[:-3]
X_test = X[-3:]
y_train = y[:-3]
y_test = y[-3:]

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_test_pred = regr.predict(X_test)

print('Coefficients: ', regr.coef_)
print('Intercept:', regr.intercept_)
print('the model is: y = ', regr.coef_, '* X + ', regr.intercept_)
# 均方误差
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred))
# r2 score，0,1之间，越接近1说明模型越好，越接近0说明模型越差
print('Variance score: %.2f' % r2_score(y_test, y_test_pred))

x0, x1 = np.meshgrid(np.asarray(X_train)[:, 0], np.asarray(X_train)[:, 1])
y = np.asarray(regr.coef_[0] * x0 + regr.coef_[1] * x1 + regr.intercept_)
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
