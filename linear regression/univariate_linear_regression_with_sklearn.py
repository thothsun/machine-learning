# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/1/13 2:33 PM
@file: univariate_linear_regression_with_sklearn.py
@desc: 单变量线性回归-使用sklearn
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 不想看到warning，添加以下代码忽略它们
warnings.filterwarnings(action="ignore", module="sklearn")

dataset = pd.read_csv('dataset.csv')
X = np.asarray(dataset.get('area')).reshape(-1, 1)
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

plt.xlabel('area')
plt.ylabel('price')
# 画训练集的散点图
plt.scatter(X_train, y_train, alpha=0.8, color='black')
# 画模型，二维空间中的一条直线
plt.plot(X_train, regr.coef_ * X_train + regr.intercept_, color='red', linewidth=1)
plt.show()
