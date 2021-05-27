# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 23:48:58 2021

@author: gbral
"""

# a naive and incomplete demonstration on how to read a *.spydata file
import pickle
import tarfile
import os

directory = r'D:\Documentos\Cesar\MASTER_PROJECT\final_project\model\data_results'

# open a .spydata file
name = 'data_3_features_logistic_ovr'
filename = name + '.pickle'

if filename in os.listdir(directory):
    real_path = os.path.join(directory, filename)
    with open(real_path, 'rb') as fdesc:
             data = pickle.loads(fdesc.read())
else:
    real_path = os.path.join(directory, name + '.spydata')
    tar = tarfile.open(real_path, "r")
    # extract all pickled files to the current working directory
    tar.extractall(path='data_results/')
    extracted_files = tar.getnames()
    for f in extracted_files:
        if f.endswith('.pickle'):
             with open('data_results/' + f, 'rb') as fdesc:
                 data = pickle.loads(fdesc.read())