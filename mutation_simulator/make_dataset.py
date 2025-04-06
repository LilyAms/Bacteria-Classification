#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:53:19 2021

@author: Lily Amsellem
"""

# File to modify and run every time you want to generate a new dataset
import numpy as np
from mutation_simulator import mutation_simulator
from data_selection import MC_species_percentages


if __name__=='__main__':
    
    dataset_params = {
        
        # Do not change 
        'raw_data_dir': '../../raw_data/new_mock_data/20210829_sample_91_species.fas',
        'out_dir': '../../simulated_datasets/',
        
        # MODIFY FROM HERE
        'experiment_name':'20210829_range_error_rates_91_species/even_distr/',
        
        # Required error rates, as a list
        # Will produce a separate fasta file for each error rate
        'error_rates': [0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6], 
        
        # Relative proportions of mutations (insertions, deletions, substitutions): 
        # if set to 'equal', the error rate will be equally divided between insertions, substitutions and deletions
        # if set to 'random', proportions of each mutation will be chosen at random
        # if set to 'ins', 'del' or 'sub', only the corresponding mutations will be applied
        'mutations_prop' : 'equal', 
        
        # The required species distribution in the sample
        # It should be a dictionary of the form 'species_name':'percentage
        # Set to None if all species should be equally distributed
        'species_distribution': None, #MC_species_percentages, 
        
        # Number of sequences to be generated in sample, PER ERROR RATE
        # For each error rate, a corresponding file is created 
        'sample_size': 15000, 
        
        # Number of species in the mock community
        # If only a subset of species are used, proportions from MC_species_percentage are adjusted
        'MC_size': 21
        }
    
    # UNCOMMENT HERE FOR CUSTOM SPECIES DISTRIBUTION
        
    # Run mutation simulator
    comm_size = [17] #[200,300]
    n_sequences = [27300]
    for size,n_seq in zip(comm_size, n_sequences):
        print('--------------------------------------')
        print("Generating dataset with {} species".format(size))
        print('\n')
        
        # if size ==100:
        #     dataset_params['raw_data_dir'] = '../../raw_data/cremated_exp_communities/20210714_sample_100_species.fas'
        # else:
        #     dataset_params['raw_data_dir'] = '../../raw_data/created_exp_communities/20210804_sample_{}_species.fas'.format(size)
        
        dataset_params['raw_data_dir'] = '../../raw_data/new_mock_data/20210829_sample_91_species.fas'
        
        dataset_params['experiment_name'] = '20210829_range_error_rates_{}_species/even_distr/'.format(size)
        dataset_params['sample_size'] = n_seq
        
        mutation_simulator(dataset_params)
        print("Dataset with {} species created".format(size))
        print('\n')
