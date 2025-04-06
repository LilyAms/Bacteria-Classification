#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:59:13 2021

@author: Lily Amsellem
"""

import scipy

model_param_grids = {
    
    'SGD_rdm':{'tfidf__ngram_range':[(3,3),(4,4),(5,5),(6,6),(7,7)],'clf__loss':['hinge', 'log'],'clf__penalty':['l2', 'elasticnet'],
           'clf__alpha': scipy.stats.uniform(0.0001,0.0005),'clf__l1_ratio': scipy.stats.uniform(loc = 0, scale = 0.2), \
               'clf__n_jobs':[-4], 'clf__random_state':[42], 'clf__early_stopping':[True]}, 
    
     'SGD_grid':{'tfidf__ngram_range':[(6,6),(7,7)],'clf__loss':['hinge', 'log'],'clf__penalty':['l2', 'l1','elasticnet'],
            'clf__alpha': [0.0001, 0.001], 'clf__early_stopping':[True], 'clf__n_jobs':[-1]},

    'RF':{'tfidf__ngram_range':[(5,5),(6,6),(7,7)],
          'clf__n_estimators':(50, 100, 200),'clf__criterion':['gini'], 
          'clf__n_jobs': [-1], 'clf__random_state':[42], 'clf__max_depth':(50,100,200)},
    
    'MLP_grid':{'clf__hidden_layer_sizes':[(100,100),(100,100,100)], 'clf__activation':['relu'], 'clf__solver':['adam'],
                'clf__alpha': [1e-5,0.0001, 1e-3], 'clf__batch_size': [32,64,200], 'clf__learning_rate_init':[0.001], 
                'clf__max_iter':[200], 'clf__shuffle':[True], 'clf__random_state' : [42], 'clf__early_stopping' : [True], 
                'clf__validation_fraction':[0.2]}
    
    }