#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 12:26:16 2021

@author: Lily Amsellem
Analyse test results: group species together, compute frequencies, and compare predicted frequencies to theoretical ones. 
"""

import re
import numpy as np
import pandas as pd
from os import path, listdir
from process_dataset_v02 import read_dataset, find_species_names, process_string, process_list

if __name__=='__main__':
    
    zymo_species = {'Akkermansia_muciniphila_CP015409.410149.411664':0.87, 
            'Bacteroides_fragilis_CP011073.5118647.5120186':9, 
            'Bifidobacterium_adolescentis_CP028341.2091809.2093347':7.95,  
            'Clostridioides_difficile_CP010888.1571905.1573413': 2.37,
            'Clostridium_perfringens_AB045283.555.2071':0.0002, 
            'Enterococcus_faecalis_AB012212.1.1517':0.0008,
            'Escherichia_coli_PSUO2':10.97, 
            'Faecalibacterium_prausnitzii_CP022479.1143644.1145162':15.96, 
            'Fusobacterium_nucleatum_CP022122.254210.255773':6.79,
            'Lactobacillus_fermentum_AVAB01000001.1563759.1565353':8.72, 
            'Methanobrevibacter_smithii_AELM01000007.2778.4273':0.060, 
            'Prevotella_corporis_AB547677.1.1479':4.51, 
            'Roseburia_hominis_CYZJ01000002.353627.355138':8.95, 
            'Salmonella_enterica_CP030026.2442764.2444316':0.008, 
            'Veillonella_rogosae_JCM_15642':14.37, 
            'Candida_albicans_SC5314_CP017630.1891236.1893022':3.11,
            'Saccharomyces_cerevisiae_AB628065.1.1497':6.35} 
    
    params = {
        # Model directories 
        'exp_dir' :  '20210828_range_error_rates_17_species/even_distr',
        'model_dir' : '20210828_sgd_model/' 
        }
    
    ref_species = process_list(list(zymo_species.keys()))
    ref_df = pd.DataFrame({'regexp_sp_label':ref_species, 
                      'theoretical_abundance': list(zymo_species.values())})
    

    
    # Load database into a dataframe
    dataset_file = '../../raw_data/large_mock_data/SILVA_138.1_SSURef_NR99_tax_silva.fasta'
    data = read_dataset(dataset_file)
    
    # load test values
    test_results_dir = '../../test_results/'
    
    full_dir = path.join(test_results_dir,path.join(params['exp_dir'],params['model_dir']))
    model_names = sorted([f for f in listdir(full_dir) if not f.startswith('.')])
    
    for model in model_names:
        model_path = path.join(full_dir, model)
        
        print('---------------------')
        print('Model: {}'.format(model))
        print('\n')
        
        # Read predictions                 
        pred_df = pd.read_csv(path.join(model_path,'predicted_abundances.csv'),index_col = False)
        
        # Look up the species full label
        labels = pred_df.label.tolist()
        full_labels = find_species_names(data, labels)
        
        # Add Species column with full label to dataframe
        pred_df = pred_df.merge(full_labels, left_on = 'label', right_on = 'Name')
        pred_df = pred_df.drop('Name', axis = 1)
        
        # Group species per name
        pred_df['regexp_sp_label'] = pred_df['Species'].apply(lambda x:process_string(x))
        species_grp = pred_df.groupby('regexp_sp_label').agg({'predicted_abundance':'sum'})

        
        df = species_grp.merge(ref_df, how = 'left', on = 'regexp_sp_label')
        
        df['abundance_diff'] = np.abs(df['theoretical_abundance'] - df['predicted_abundance'])
        df ['relative_diff'] = np.abs((df['predicted_abundance'] - df['theoretical_abundance'])/df['theoretical_abundance'])
                
        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0'], axis = 1)
        
        pred_df.to_csv(path.join(model_path,'full_preds.csv'))
        df.to_csv(path.join(model_path,'pred_comparisons.csv'))
                
        rmse = np.sqrt(np.mean((df['theoretical_abundance'] - df['predicted_abundance'])**2))
        mae = np.mean(np.abs(df['theoretical_abundance'] - df['predicted_abundance']))
        
        
        print('Model RMSE: {}'.format(rmse))
        print('Model MAE: {}'.format(mae))
        print('\n')
    
