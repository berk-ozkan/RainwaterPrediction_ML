# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:44:09 2023

@author: Berk Özkan
"""
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Read the dataset from a CSV file
df = pd.read_csv("C:/Users/DELL/Desktop/kod/Dataset.csv")

# Drop rows with missing values
df = df.dropna()
#print(df)

""" 
BB: Daily Vapor Pressure
GMS: Daily Minimum Temperature
GOAB: Daily Average Actual Pressure
GONN: Daily Average Relative Humidity
GTY: Daily Total Precipitation
""" 
# Selecting features (X) and target variable (y)
X = df[['BB','GMS', 'GOAB', 'GONN']]
y = df['GTY']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#print(X_train, X_test, y_train, y_test)

# Standardize the features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# K-Nearest Neighbors (KNN) Regression
from sklearn.neighbors import KNeighborsRegressor
# Create KNN model with 7 neighbors and Minkowski distance metric
knn = KNeighborsRegressor(n_neighbors = 7, metric='minkowski')
# Train the KNN model with the standardized training data
knn.fit(X_train, y_train)
# Predict the target variable on the standardized test data
y_pred = knn.predict(X_test)
# Calculate Mean Absolute Error (MAE) for KNN
mae = mean_absolute_error(y_test, y_pred)
print("KNN Mean Absolute Error: ", mae)
# Calculate R-squared (R2) score for KNN
r2 = r2_score(y_test, y_pred)
print("KNN R2 Score:", r2)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
# Create Decision Tree model with a maximum depth of 4
dtr = DecisionTreeRegressor(max_depth=4)
# Train the Decision Tree model with the standardized training data
dtr.fit(X_train, y_train)
# Predict the target variable on the standardized test data
y_pred1 = dtr.predict(X_test)
# Calculate Mean Absolute Error (MAE) for Decision Tree
mae1 = mean_absolute_error(y_test, y_pred1)
print("\nDecision Tree Mean Absolute Error: ", mae1)
# Calculate R-squared (R2) score for Decision Tree
r2_1 = r2_score(y_test, y_pred1)
print("Decision Tree R2 Score:", r2_1)

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
# Create Random Forest model with a maximum depth of 3
rfr = RandomForestRegressor(max_depth=3)
# Train the Random Forest model with the standardized training data
rfr.fit(X_train, y_train)
# Predict the target variable on the standardized test data
y_pred2 = rfr.predict(X_test)
# Calculate Mean Absolute Error (MAE) for Random Forest
mae2 = mean_absolute_error(y_test, y_pred2)
print("\nRandom Forest Mean Absolute Error: ", mae2)
# Calculate R-squared (R2) score for Random Forest
r2_2 = r2_score(y_test, y_pred2)
print("Random Forest R2 Score:", r2_2)

# Support Vector Regression (SVR)
from sklearn.svm import SVR
# Create SVR model with a polynomial kernel of degree 5
svr = SVR(kernel = 'poly', degree = 5)
# Train the SVR model with the standardized training data
svr.fit(X_train, y_train)
# Predict the target variable on the standardized test data
y_pred3 = svr.predict(X_test)
# Calculate Mean Absolute Error (MAE) for SVR
mae3 = mean_absolute_error(y_test, y_pred3)
print("\nSVR Mean Absolute Error: ", mae3)
# Calculate R-squared (R2) score for SVR
r2_3 = r2_score(y_test, y_pred3)
print("SVR R2 Score:", r2_3)
