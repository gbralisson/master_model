# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:01:23 2021

@author: gbral
"""

# Logistic Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data_3_scenarios.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Transforming outcome
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
lable_mapping = {k:v for k,v in enumerate(le.classes_)}

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Prepare model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
model = OneVsOneClassifier(LogisticRegression())

# # Prepare the cross-validation procedure
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# cv = KFold(n_splits=10, random_state=1, shuffle=True)
# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# print('KFold Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Prepare the repeated cross-validation procedure
# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import cross_val_score
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# print('Repeated KFold Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_comp = pd.DataFrame(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)