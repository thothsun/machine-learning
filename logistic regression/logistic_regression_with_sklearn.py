# encoding: utf-8
"""
@author: suns
@contact: sunshuai0518@gmail.com
@time: 2019/7/1 3:22 PM
@file: logistic_regression_with_sklearn.py
@desc:
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer = datasets.load_breast_cancer()

X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print('测试集准确率：', accuracy_score(y_test, predictions))

print('测试集的第10个的预测值：', model.predict([X_test[9]]))
print('测试集的第10个的真实值：', y_test[9])
