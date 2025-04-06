#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:15:51 2021

@author: Lily Amsellem
"""

import sys
sys.path.insert(0,'../data_processing/')

from os import path, listdir
import numpy as np
import pandas as pd
import re
from data_generation import parse_fasta
from model_class import grid_search_cv, grid_search
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def make_dataset(files):
    
    seq_df = pd.DataFrame()
    
    for file in files:
        # Get error rate from file name
        err_rates = [float(s) for s in re.findall('[\d]*[.][\d]+', file)]
        err_rate = round(np.sum(err_rates),2)
        
        
        # Load data
        file_df = parse_fasta(file)
        file_df['Error Rate'] = err_rate
        
        seq_df = seq_df.append(file_df)
    
    return seq_df

def model_selection(cv = False):
    
    best_score = -np.inf
    final_params = None
    
    models = [('LogReg', LogisticRegression()), 
           ('SVC', SVC()),
           ('RandomForest', RandomForestClassifier()), 
           ('AdaBoost', AdaBoostClassifier()),
           ('MLP', MLPClassifier(hidden_layer_sizes= (150,100,50), max_iter = 500))]
    
    param_grids = [('LogReg',
                    {'solver':('newton-cg', 'lbfgs'), 'C':(1, 0.5,0.1), 'max_iter':[500]}),
                    ('SVC', 
                     {'C':(1,0.5,0.1), 'kernel':('rbf', 'linear')}),
                    ('RandomForest', 
                     {'n_estimators':(100,200), 'criterion':('gini','entropy')}),
                    ('AdaBoost',
                     {'base_estimator':(DecisionTreeClassifier(max_depth = 2), DecisionTreeClassifier(max_depth = 3)), 'n_estimators':(200,250)}),
                    ('MLP', 
                     {'hidden_layer_sizes':((100,), (100,100,)), 'max_iter':(200,500)})]
    
    # ------------------------- RUN GRID SEARCH FOR EACH MODEL ---------------------------
    
    for idx, model in enumerate(models):
        name, clf = model
        param_grid = param_grids[idx][1]
        if cv:
            score, best_params = grid_search_cv(clf, X_train, y_train, param_grid, kmer_range, nfolds = nfolds, verbose = False)
        else:
            score, best_params = grid_search(clf, X_train, y_train, param_grid, kmer_range, val_size = 0.2)
        print("Best score for {} model: {}".format(name,score))
        print('\n')
    
        if score > best_score:
            best_model = model
            best_score = score
            final_params = best_params
            
    print('Best model: {}'.format(best_model[0]))
    if cv:
        print('{}-fold CV Score: {}'.format(nfolds, best_score))
    else:
        print('Best Validation Score: {}'.format(best_score))
    print('\n')
        
    return best_score, final_params
    

if __name__ == '__main__':
    
    np.random.seed(0)
    
    
    # SET PARAMETERS HERE
    loc_dir = '../../simulated_datasets/'
    exp_dir = '20210618_eq_prop_exp'
    #model_name = 'svc_model_error_range'
    
    err_rate = 'range'
    
    test_size = 0.2
    nfolds = 4
    kmer_ranges = [(4,4),(5,5), (6,6), (7,7)]
    
    data_dir = path.join(loc_dir, exp_dir)
    record_paths = sorted([f for f in listdir(data_dir) if not f.startswith('.')
                          and f.startswith('in') or f.startswith('nb_it')])
    
    record_paths = [path.join(data_dir, path.join(rec_path,"sequences.fst"))  for rec_path in record_paths]
    
    # data = make_dataset(record_paths)
        
    # X = data['Sequence'].values
    # y = data['Name'].values
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
            
    
    
    # #-------------------------------------- SPLIT DATASET -------------------------------------
    # idx = np.arange(len(X)//2)
    # np.random.shuffle(idx)
    # X = X[idx]
    # y = y[idx]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
    
    # print(np.unique(y_train, return_counts = True))
    
    cv = False
    #val_scores = []
    # for k in kmer_ranges:
    #     kmer_range = k
    #     print('Running model Selection on {}-mer model'.format(kmer_range[0]))
    #     print('-------------------------------')
    #     best_val_score, best_params = model_selection(cv)
    #     val_scores.append(best_val_score)
    #     print('-------------------------------')
    #     print('\n')
    
    
    # ---- FOR SEVERAL ERROR RATES---------------
    error_rates = []
    val_scores = []
    kmer_range = (6,6)
    for record_path in record_paths:
        err_rates = [float(s) for s in re.findall('[\d]*[.][\d]+', record_path)]
        err_rate = round(np.sum(err_rates),2)
        error_rates.append(err_rate)
        
        
        seq_df = parse_fasta(record_path)
        X = seq_df['Sequence'].values
        y = seq_df['Name'].values
        
        #-------------------------------------- SPLIT DATASET -------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)
        
        

        print('Running model Selection for error rate: {}'.format(err_rate))
        print('-------------------------------')
        best_val_score, best_params = model_selection(cv)
        val_scores.append(best_val_score)
        print('-------------------------------')
        print('\n')
        
        
        
    
    