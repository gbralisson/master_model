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
name = 'data_5_features_logistic_ovo'
filename = name + '.pickle'

if filename in os.listdir(directory):
    with open('data_results/' + filename, 'rb') as fdesc:
             data = pickle.loads(fdesc.read())
else:    
    tar = tarfile.open('data_results/' + name + '.spydata', "r")
    # extract all pickled files to the current working directory
    tar.extractall(path='data_results/')
    extracted_files = tar.getnames()
    for f in extracted_files:
        if f.endswith('.pickle'):
             with open('data_results/' + f, 'rb') as fdesc:
                 data = pickle.loads(fdesc.read())