#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:41:42 2021

@author: Lily Amsellem
"""

import sys
sys.path.insert(0,'../data_processing/')

import argparse
from os import path, listdir, makedirs
import numpy as np
from full_model import load_clf
from data_generation import parse_fasta
import pandas as pd
from process_dataset_v02 import read_dataset





def get_data_from_files(file_path, format = 'fasta'):
    """
    Load a list of fasta or fastq files into a dataframe. 

    Parameters
    ----------
    file_path : list
        list of files either in fasta or fastq format to be loaded in.
    format : string, optional
        File format for loading using BioPython SeqIO libray. The default is 'fasta'.

    Returns
    -------
    data : pandas dataframe
        Dataframe containing the columns 'Name' and 'Sequence', for the species and the corresponding DNA sequence.

    """
    
    file_list = sorted([f for f in listdir(file_path)])
    
    data = pd.DataFrame()
    
    for file in file_list:
        file_df = parse_fasta(path.join(file_path, file), format = format)
        data = data.append(file_df, ignore_index = True)
    
    return data

    

def compute_abundances(y_pred, lb):
    """
    Compute abundances of species (as percentages) from a numpy array of numeric labels, each label corresponding to a species. 

    Parameters
    ----------
    y_pred : numpy array
        vector of predictions (species numeric label).
    lb : LabelEncoder object from sklearn.
        Label encoder to "decode" the labels encoded into numbers.

    Returns
    -------
    labels : numpy array 
        Original labels for the vector of numeric labels y_pred.
    abundances : numpy array
        For each label, the corresponding frequency in the vector y_pred.

    """
    
    labels, abundances = np.unique(y_pred, return_counts = True)
    labels = lb.inverse_transform(labels)
    abundances = abundances*100/np.sum(abundances)
    
    return labels, abundances

def save_predictions(save_dir, y_pred, lb):
    """
    Save the vector of predictions to a text file. 
    Also save a dataframe containing the species and their abundances to a csv. 

    Parameters
    ----------
    save_dir : string
       Directory where to save the prediction results.
    y_pred : numpy array
        numpy array of predictions.
    lb : LabelEncoder object from sklearn. 
        To decode the encoded labels in y_pred, and find the corresponding original labels. 

    Returns
    -------
    pred_df : pandas dataframe
        DataFrame containing 'label' and 'predicted_abundance' columns.

    """
    
    try:
        makedirs(save_dir)
    except OSError:
        pass

    
    # Save raw predictions
    np.savetxt(path.join(save_dir,'predicted_labels.txt'), y_pred)
    
    # Compute abundances and save them in a dataframe
    labels, abundances = compute_abundances(y_pred,lb)   
    pred_df = pd.DataFrame({'label':labels, 'predicted_abundance':abundances})
    pred_df.to_csv(path.join(save_dir, 'predicted_abundances.csv'), index = False)
    
    return pred_df



if __name__=='__main__':

    np.random.seed(0)
    
    
    # SET MODEL PARAMETERS HERE
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('-exp_file_name', help = 'Name of experiment directory in simulated_datasets/')
    
    argparser.add_argument('-model_dir', help = 'Model directory, usually of the form Y%M%D_model_name')
        

    args = argparser.parse_args()
    exp_dir = args.exp_file_name
    model_dir = args.model_dir 

    
    params = {
        
        # Model directories
        'loc_dir': '../../saved_models',    
        'exp_dir' :  exp_dir, 
        'model_dir' : model_dir,
        
        # Data-related directories
        'data_dir' : '../../simulated_datasets/'     
        }

    
    # Load database into a dataframe
    dataset_file = '../../raw_data/large_mock_data/SILVA_138.1_SSURef_NR99_tax_silva.fasta'
    data = read_dataset(dataset_file)
    
    # Load data to be tested from test files
    test_files_path = '../../raw_data/barcode22/'
    test_data = get_data_from_files(test_files_path, format = 'fastq')
    
    models_path = path.join(params['loc_dir'], path.join(params['exp_dir'], params['model_dir']))
    model_names = sorted([f for f in listdir(models_path) if not f.startswith('.')])

    # Make predictions for each model
    for model_name in model_names:
        model_path = path.join(models_path, model_name)
        
        model, lb = load_clf(model_path)
        
        print('--------------------------')
        print("Testing model {}".format(model_name))
        print('\n')
        
        X_test = test_data['Sequence'].values
        
        # Predict test labels
        print("Predicting on test data ...")
        y_pred = model.predict(X_test)
        labels, abundances = compute_abundances(y_pred, lb)
    
        
        # Save predictions
        test_results_dir = '../../test_results/'
        save_dir = path.join(test_results_dir,path.join(params['exp_dir'],path.join(params['model_dir'], model_name)))    
        pred_abundances = save_predictions(save_dir, y_pred, lb)
    
        print('Test predictions saved.')
        print('\n')
