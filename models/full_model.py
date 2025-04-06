#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:50:39 2021

@author: Lily Amsellem
"""

import sys
sys.path.insert(0,'../data_processing/')

import time
from os import listdir, path, makedirs
import argparse
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import seaborn as sns
import datetime
import joblib
from joblib import dump, load

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from plot_learning_curve import plot_learning_curve 

from model_params import model_param_grids
from data_generation import parse_fasta



def write_report(model_dir, clf_report, model_results, cm = None, training = False, plot_conf_matrix = False):
    """
    Write train/test report of fitted model

    Parameters
    ---------- 
    model_dir : str, path to the directory where the report should be saved
    clf_report: classification_report from sklearn.metrics
    model_results : dict, optional. 
        Dict containing some information about the model, including train/test scores, test size, model params.... 
        The default is None.
    training : bool, optional
        If set to True, will save reports with 'train' tag, otherwise with 'test. The default is False.
    cm: a confusion matrix to save, and plot if plot_conf_matrix is True. 
    plot_conf_matrix : bool, optional
       If set to True, it will display the confusion matrix. The default is False.

    Returns
    -------
    None.

    """
    
    report_dir = path.join(model_dir,'train_report.txt') if training else path.join(model_dir,'test_report.txt')
            
    with open(report_dir,'w') as rep_file:
        if model_results is not None:
            for k,v in model_results.items():
                rep_file.write('{} : {}'.format(k,str(v)))
                rep_file.write('\n')
            rep_file.write('\n')

            
        rep_file.write('Classification Report: ')
        rep_file.write('\n')
        rep_file.write(clf_report)
        rep_file.write('\n')
        if cm is not None:
            rep_file.write('Confusion Matrix')
            rep_file.write('\n')
            np.savetxt(rep_file, cm, fmt='%1.3f')
       
        
    if plot_conf_matrix:
        disp = sns.heatmap(cm, square = True, annot = False)
        save_dir = 'train_conf_matrix.png' if training else 'test_conf_matrix.png'
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted Class')
        plt.title('Confusion matrix for the classification task')
        #fig.savefig(path.join(model_dir, save_dir))
        plt.savefig(path.join(model_dir, save_dir))
        plt.clf()
        plt.close()
    
    
def save_model(model_dir, model, label_enc = None):
    """
    Save the fitted classifier self.clf, as well as the preprocessors and label encoder. 

    Parameters
    ----------
    model_dir : str, path where the model should be saved. 
    model: an sklearn pipeline or model object which has been fitted
    label_enc: a fitted sklearn label encoder object

    Returns
    -------
    None.

    """
    
    try:
        makedirs(model_dir)
    except OSError:
        pass
    
    joblib.dump(model, path.join(model_dir, 'model.joblib'))
    
    if label_enc is not None:
        joblib.dump(label_enc, path.join(model_dir, 'label_encoder.joblib'))

    
                
def load_clf(model_dir):
    """
    Load a previously saved model. 

    Parameters
    ----------
    model_dir : string, directory where the model is saved

    Returns
    -------
    model: the loaded model
    lb : the loaded label encoder to find the original labels from encoded ones. 

    """
    lb_path = path.join(model_dir, 'label_encoder.joblib')
    
    if path.isfile(lb_path):
        lb = joblib.load(lb_path)
    
    model = joblib.load(path.join(model_dir, 'model.joblib'))
    
    return model, lb

def main(params, X, y):
    """
    

    Parameters
    ----------
    params : dict
        Dictionary of parameters for the run.
    X : Numpy array
        Numpy array containing the training data.
    y : Numpy array
        Numpy array containing the training labels.
    
    Trains the model as set in params, and saves results and learning curves. 

    Returns
    -------
    None.

    """
    
    #-------------------------------------- SPLIT DATASET -------------------------------------
    
    # Train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = params['test_size'], random_state = 42)
    
    # ------------------------------------ ENCODE LABELS --------------------------------------
    lb = LabelEncoder()
    y_train_tr = lb.fit_transform(y_train)
    y_test_tr = lb.transform(y_test)
    
    if params['feature_selection'] is not None:
            
        if params['feature_selection']=='chi2':
            params['feature_selector'] =  SelectKBest(chi2)
    
        if params['feature_selection']=='mutual_info':
            params['feature_selector'] =  SelectKBest(mutual_info_classif)
        
        elif params['feature_selection'] == 'SelectFromModel':
             params['feature_selector'] = SelectFromModel(SGDClassifier())
             
        model = Pipeline([('tfidf', TfidfVectorizer(analyzer = 'char', ngram_range = params['kmer_range'], use_idf = True)),\
                          ('normalizer', Normalizer()),
                          ('f_selector', params['feature_selector']),
                          ('clf', params['clf'])])
    else:
        
        if params['model_name']=='MLP':
            model = Pipeline([('one_hot_enc',OneHotEncoder()),
                              ('clf', params['clf'])])
            
        else:
            model = Pipeline([('tfidf', TfidfVectorizer(analyzer = 'char', ngram_range = params['kmer_range'], use_idf = True)),\
                              ('normalizer', Normalizer()),
                              ('clf', params['clf'])])
    
    # -------------------------------------- RUN GRIDSEARCH FOR MODEL SELECTION -------------------------------------
    if params['grid_search'] is not None:
        start = time.time()
        
        # Train and Val splits 
        sss = StratifiedShuffleSplit(n_splits = params['cv'], test_size = params['val_size'], random_state = 42)
        
        
        if params['grid_search'] == 'random':
            print('Running Randomized Hyperparameter Search...')
            print('\n')
            hyp_search = RandomizedSearchCV(model, param_distributions = params['param_grid'],\
                                            n_jobs = -2, n_iter = params['grid_search_iter'], cv = sss, \
                                            return_train_score = True, verbose = 1)
        elif params['grid_search'] == 'grid':
            print('Running Hyperparameter Grid Search...')
            print('\n')
            
            hyp_search = GridSearchCV(model, param_grid = params['param_grid'],\
                                            n_jobs = -2,cv = sss, \
                                            return_train_score = True, verbose = 1)
                      
        
        hyp_search.fit(X_train, y_train_tr)
        
        print('Grid search done in {}.'.format(time.time()- start))
        print('\n')
        
        best_model, best_params = hyp_search.best_estimator_, hyp_search.best_params_
        
        print('Found best params: {} '.format(best_params))
        print('\n')
        
        # Save training stats
        cv_results = hyp_search.cv_results_
        best_comb_idx = np.where(cv_results['rank_test_score']==1)[0][0]
        train_score = cv_results['mean_train_score'][best_comb_idx]
        val_score = cv_results['mean_test_score'][best_comb_idx]
        fit_time = cv_results['mean_fit_time'][best_comb_idx]
            
                
    else:
        
        # Train and Val splits 
        sss = StratifiedShuffleSplit(n_splits = params['cv'], test_size = params['val_size'], random_state = 42)
        
        best_model = model
        best_params = params['clf'].get_params()
        
        print('Training model...')
        print('\n')
        start = time.time()        
        best_model.fit(X_train, y_train_tr)
        fit_time = time.time()-start
        train_score = best_model.score(X_train, y_train_tr)
        val_score = None
    
    print('Training Accuracy: {}'.format(train_score))
    print('\n')
    if val_score is not None:
        print("Validation Accuracy: {}".format(val_score))
        print('\n')
        
    y_train_pred = best_model.predict(X_train)
    train_clf_report = classification_report(y_train_tr, y_train_pred)
    train_cm = confusion_matrix(y_train_tr, y_train_pred)

    # Test the model
    print("Testing model...")
    test_score = best_model.score(X_test, y_test_tr) 
    print("Test Accuracy: {}".format(test_score))

    y_test_pred = best_model.predict(X_test)
    test_clf_report = classification_report(y_test_tr, y_test_pred)
    test_cm = confusion_matrix(y_test_tr, y_test_pred)
        
    # Save report on training and testing
    model_results = {'Model': best_model['clf'],'Test size': params['test_size'], 'kmer_range': best_model['tfidf'].ngram_range,
                          'Train accuracy': train_score,'Validation accuracy': val_score, 
                          'Test score': test_score, 'Fit time': fit_time}  
    
    
    # -------------------------------------- PICK AND SAVE BEST MODEL -------------------------------------
    # Save model and write results summary
    model_dir = path.join('../../saved_models',path.join(params['exp_dir'],path.join(params['model_dir'],params['model_name'])))    
    if params['model_name'] is not None:
        print('Saving Model...')
        print('\n')
        save_model(model_dir, best_model, lb)
        write_report(model_dir, train_clf_report, model_results, train_cm, training = True, plot_conf_matrix = True)
        write_report(model_dir, test_clf_report, model_results, test_cm, training = False, plot_conf_matrix = True)
        print('Model saved.')
        print('\n')
    
    
    if params['save_learning_curves']:
        print('Plotting Learning Curves...')
        print('\n')
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        plot_learning_curve(best_model, 'Learning Curves', X_train, y_train_tr, axes=axes, ylim=(0,1.05),
                            cv=sss, n_jobs = -2, train_sizes=np.linspace(0.5, 1.0, 5), save_to_file = path.join(model_dir,'model_performance.png'))

if __name__=='__main__':
    
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('-exp_file_name', help = 'Name of experiment directory in simulated_datasets/')
    
    argparser.add_argument('-model_name', help = 'Name of the model, it will me saved as %Y%m%d_model_name')
    
    argparser.add_argument('-param_grid', help = 'Dict of parameters for hyperparameter grid search, to find in file model_params.py')


    args = argparser.parse_args()

    today = datetime.datetime.today().strftime('%Y%m%d_')
    model_name = args.model_name
    model_dir = today + model_name
    exp_dir = args.exp_file_name
    
    

    params = {
        
        'loc_dir':'../../simulated_datasets',
        
        # Name of experiment : subdirectory in loc_dir where the experiment data files are contained
        'exp_dir' : exp_dir,
        
        # Model name for saving in saved_models
        # If None, model won't be saved
        'model_dir': model_dir,
        
        # Name of the model : 'sgd_model' or 'random_forest'
        'model_name' : model_name, 
        
        
        # Proportion of total dataset used for validation, and testing
        'val_size': 0.2,
        
        'test_size' : 0.2,
        
        # kmer range : kmer length for feature extraction with Th-idf
        'kmer_range' : (6,6),
        
        # Classifier to train
        'clf': RandomForestClassifier(), 
        
        # Number of folds for cross-validation
        # Default to 1 for usual train/val splits 
        'cv' : 1,
        
        'save_learning_curves': True,
        
        # SET TO 'random', 'grid', or None
        # If 'random', perform randomized hyperparameter search
        # If 'grid', perform grid search over a list of parameter combinations
        # If None, no grid search is performed, the parameters passed to params['clf'] are used
        'grid_search': 'grid',
        
        # Number of param combinations to try out
        'grid_search_iter': 10,
        
        # Grid of params to search during grid search
        'param_grid': model_param_grids[args.param_grid]
        }
    
    data_dir = path.join(params['loc_dir'], params['exp_dir'])
    record_paths = sorted([f for f in listdir(data_dir) if not f.startswith('.')
                          and (f.startswith('nb_it') or f.startswith('in_'))])
        
    record_paths = [path.join(data_dir, path.join(rec_path,"sequences.fst"))  for rec_path in record_paths]
    
    for record_path in record_paths:
        err_rates = [float(s) for s in re.findall('[\d]*[.][\d]+', record_path)]
        err_rate = round(np.sum(err_rates),2) 
        params['model_name'] = model_name + '_err_rate_{}'.format(err_rate)
        
        if err_rate >=0.2 and err_rate <=0.6:
            seq_df = parse_fasta(record_path)
            X = seq_df['Sequence'].values
            y = seq_df['Name'].values
            
            print('Running model for error rate: {}'.format(err_rate))
            print('-------------------------------')
            print('\n')
            main(params, X, y)
        
    