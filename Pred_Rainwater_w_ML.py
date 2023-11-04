# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:44:09 2023

@author: Berk Özkan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#dataset
df = pd.read_csv("C:/Users/DELL/Desktop/kod/Dataset.csv")


df = df.dropna()

#print(df)

X = df[['BB','GMS', 'GOAB', 'GONN']]

#print(X)

y = df['GTY']
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#print(X_train, X_test, y_train, y_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 7, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("KNN Mean Absolute Error: ", mae)
r2 = r2_score(y_test, y_pred)
print("KNN R2 Score:", r2)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth=4)
dtr.fit(X_train, y_train)
y_pred1 = dtr.predict(X_test)
mae1 = mean_absolute_error(y_test, y_pred1)
print("\nDecision Tree Mean Absolute Error: ", mae1)
r2_1 = r2_score(y_test, y_pred1)
print("Decision Tree R2 Score:", r2_1)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth=3)
rfr.fit(X_train, y_train)
y_pred2 = rfr.predict(X_test)
mae2 = mean_absolute_error(y_test, y_pred2)
print("\nRandom Forest Mean Absolute Error: ", mae2)
r2_2 = r2_score(y_test, y_pred2)
print("Random Forest R2 Score:", r2_2)

from sklearn.svm import SVR
svr = SVR(kernel = 'poly', degree = 5)
svr.fit(X_train, y_train)
y_pred3 = svr.predict(X_test)
mae3 = mean_absolute_error(y_test, y_pred3)
print("\nSVR Mean Absolute Error: ", mae3)
r2_3 = r2_score(y_test, y_pred3)
print("SVR R2 Score:", r2_3)
