# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:11:21 2021

@author: gbral
"""

# SVM

# Importing the libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

directory = r'D:\Documentos\Cesar\MASTER_PROJECT\final_project\model\data'
file = 'data_5_features.csv'
path = os.path.join(directory, file)

# Importing the dataset
dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Transforming features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Transforming outcome
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
lable_mapping = {k:v for k,v in enumerate(le.classes_)}

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,2:] = sc.fit_transform(X_train[:,2:])
X_test[:,2:] = sc.transform(X_test[:,2:])

# Prepare model
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
model = OneVsOneClassifier(SVC())

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
fig, ax = plt.subplots(figsize=(7,4))
cm = confusion_matrix(y_test, y_pred)
xlabel = ['BT', 'CB', 'CTV', 'ET', 'IB', 'IV', 'NP', 'PB', 'RB', 'RTB', 'SB', 'SP', 'SWT', 'TV']
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.0%', cmap='Blues', ax=ax, vmin=0, linewidths=1, xticklabels=xlabel, yticklabels=xlabel)
plt.savefig('test.pdf')
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=None)
rec = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
