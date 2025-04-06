#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:48:58 2021

@author: Lily Amsellem
"""

# Run simulation code


import json
import datetime
import os
import numpy as np
from mutations import generate_mutation_data
from data_selection import adjust_proportions


def load_config(config_dir):
    """
    Load json config file with the arguments for the mutation simulator run

    Parameters
    ----------
    config_name : name of the YAML config file

    Returns
    -------
    config : loaded configuration arguments for the mutation simulator run from the config file

    """
    with open(config_dir, 'r') as f:
        config = json.load(f)
        
    return config

def get_err_rates(err, mutations_prop = 'equal'):
    """
    Generate list of error rates for insertions, deletions and substitutions from a single error rate err. 
    The sum of the individual error rates add up to err. 

    Parameters
    ----------
    err : Total error rate, including insertions, substitutions, deletions
    mutations_prop : str, optional
        How the error rate should be split between insertions, deletions and substitutions. 
        The default is 'equal' for equal proportions.
        Other possibilities are 'random' for random split, 'ins'/'del'/'sub' for insertion/del/sub only. 

    Returns
    -------
    list of the form [ins_rate, del_rate, sub_rate]

    """
    
    if mutations_prop == 'equal':
        fix_err = round(err/3,3)
        return [fix_err,fix_err,fix_err]
    
    elif mutations_prop == 'random':
        error_rates = np.random.rand(3)
        error_rates = list(np.around(error_rates/np.sum(error_rates)*err, decimals = 3))
        return error_rates
    
    elif mutations_prop == 'ins':
        return [err,0,0]
    
    elif mutations_prop == 'del':
        return [0,err,0]
    
    elif mutations_prop == 'sub':
        return [0,0,err]
    
def draw_sample(species_distribution, MC_size, sample_size):
    """
    Draw a sample of species from a dataframe according to species_distribution, without replacement. 

    Parameters
    ----------

    species_distribution : Dictionary of the form species_name: species percentage in sample. 
    
    sample_size: the total size of the sample to be generated. 

    Returns
    -------
    num_species : a numpy array containing the number of sequences to be generated for each species
    """
    
    # Adjust percentages according to the species present in the sample 
    adj_sp_props = adjust_proportions(species_distribution, MC_size)
    
    p = np.array(list(adj_sp_props.values()))
    
    species = np.random.choice(np.arange(len(adj_sp_props)), size = sample_size, replace = True, p = p)
    n_species = np.unique(species, return_counts = True)
    
    num_species = np.zeros(len(p))
    for sp,nb in zip(*n_species):
        num_species[sp] = nb
        
    return num_species
        
def mutation_simulator(config):
    """
    Main function to generate new sequences with errors from a config file with parameters. 
    This function is run directly from the make_dataset.py file

    Parameters
    ----------
    config : dict containing all the required parameters to run the mutation simulator. 

    Returns
    -------
    None.

    """
    
    # Save configuration in a YAML file
    out_dir = os.path.join(config['out_dir'], config['experiment_name'])
    
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    
    json_dir = os.path.join(out_dir, 'config.json')
    with open(json_dir, 'w') as json_file:
        json.dump(config, json_file)
    
    # Run mutation simulator
    np.random.seed(0)
    
    if config['species_distribution'] is not None:
        species_distribution = draw_sample(config['species_distribution'], config['MC_size'], config['sample_size'])
    
    else:
        species_distribution = None
    
    for err in config['error_rates']:
        error_rates = get_err_rates(err, mutations_prop = config['mutations_prop'])
        generate_mutation_data(config['raw_data_dir'], error_rates, species_distribution, config['sample_size'], out_dir)
        

# if __name__ == '__main__':
    
#     sim_data_path = '../../simulated_datasets/'
#     config_name = 'config.yaml'
#     dflt_exp_name = datetime.datetime.today().strftime('%Y_%m_%d')
    
#     argparser = argparse.ArgumentParser()
    
#     argparser.add_argument('-exp_file_name', default = dflt_exp_name, help = 'Name of the experiment file containing the YAML config file for the simulator run')
    
#     args = argparser.parse_args()
    

#     config_path = os.path.join(sim_data_path, args.exp_file_name)

#     config = load_config(config_path, config_name)
    


    
    
    
    
    
