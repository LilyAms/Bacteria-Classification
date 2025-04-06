#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 11:15:59 2021

@author: Lily Amsellem
"""
import re
import pandas as pd
import numpy as np
from data_generation import parse_fasta

MC_species_percentages = {'Akkermansia_muciniphila_CP015409.410149.411664':0.87, 
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

new_species = {'Bacteroides_caccae':0,
       'Bacteroides_dorei':0, 
       'Bacteroides_ovatus': 0, 
       'Bacteroides_plebeius': 0,
       'Bacteroides_vulgatus':0, 
       'Bifidobacterium_bifidum': 0,
       'Bifidobacterium_breve':0, 
       'Bifidobacterium_catenulatum':0,
       'Bifidobacterium_longum':0, 
       'Enterococcus_caccae':0,
       'Enterococcus_faecium':0, 
       'Parabacteroides_distasonis':0,
       'Parabacteroides_johnsonii':0, 
       'Parabacteroides_merdae':0,
       'Roseburia_faecis':0, 
       'Roseburia_intestinalis':0}


def make_dataset(files, sample_size = None):
    """
    Group several fasta files into a single dataset (pandas DataFrame).

    Parameters
    ----------
    files : list, list of fasta files to be concatenated together in the output DataFrame.
    sample_size : int, number of sequences to be sampled from each file. Each species will be equally sampled.
    Useful when each file contains too many sequences for training. The default is None.

    Returns
    -------
    seq_df : pandas dataframe
        DataFrame containing the columns 'Name' for sp.

    """
    
    seq_df = pd.DataFrame()
    
    for file in files:
        # Get error rate from file name
        err_rates = [float(s) for s in re.findall('[\d]*[.][\d]+', file)]
        err_rate = round(np.sum(err_rates),2)
        
        # Load data
        file_df = parse_fasta(file)
        file_df['Error Rate'] = err_rate
        
        if sample_size:
            species_grp = file_df.groupby('Name')
            species_samples = species_grp.apply(lambda x:x.sample(n = sample_size//len(species_grp), replace = False, random_state = 42))
            seq_df = seq_df.append(species_samples, ignore_index = True)
        else:
            seq_df = seq_df.append(file_df, ignore_index = True)
    
    return seq_df


def adjust_proportions(species_percentages, MC_size):
    """
    Compute the reweighted percentages of species in our sample to mimick the ones of the mock community, 
    if not all species are present in our sample. 

    Parameters
    ----------
    species_percentages : Dict with species names as keys and percentage in mock community as value
    MC_size : Total size of mock community. 

    Returns
    -------
    proportions : Dict containg the reweighted percentages of each species in our sample 
    according to the percentages in the mock community. 

    """
    
    adj_sp_props = {}
    
    for sp, perc in species_percentages.items():
        adj_sp_props[sp] = perc
        #new_perc = perc*len(species_percentages)/MC_size
        #adj_sp_props[sp] = round(new_perc, 5)
    
    s = sum(list(adj_sp_props.values()))
    for sp in adj_sp_props.keys():
        adj_sp_props[sp]/=s
    
    return adj_sp_props

def add_species(species_percentages, new_species):
    """
    Generate random species proportions to simulate an uneven distribution of the species. 
    Species percentages are the species with available percentages. new_species are the new species for which 
    a random percentage should be generated. 

    Parameters
    ----------
    species_percentages : dict, of the form 'species':percentage. 
    new_species : dict, of the form 'species':0.

    Returns
    -------
    new_species_perc: a new dict, containing the species from species_percentages in addition to new_species, of the 
    form 'species':percentage. 
    Percentages sum to 100. 

    """
    remainder = 100 - np.sum(list(species_percentages.values()))
    
    new_species_perc = np.random.choice(np.arange(100), size = len(new_species))
    new_species_perc = (new_species_perc/np.sum(new_species_perc))*remainder
    
    return new_species_perc

# def draw_sample(data, species_percentages, MC_size, sample_size):
#     """
#     Draw a sample of species from a dataframe according to species_percentages, without replacement. 

#     Parameters
#     ----------
#     data : A dataframe containing DNA sequences from various species. 
#     It should have a 'Name column with the species name. 
#     species_percentages : Dictionary of the form species_name: species percentage in sample. 
    
#     sample_size: the size of the sample to be drawn according to the species_proportions from the data. 
#     sample_size should be smaller than data size

#     Returns
#     -------
#     adj_data : DataFrame with adjusted proportions of each species. 

#     """
    
#     adj_data = pd.DataFrame()
    
#     adj_sp_props = adjust_proportions(species_percentages, MC_size)
    
#     p = np.array(list(adj_sp_props.values()))
    
#     if sample_size>len(data):
#         raise Exception("Sample size ({}) is greater than number of samples in data set ({})".format(sample_size, len(data)))
    
#     species = np.random.choice(np.arange(len(adj_sp_props)), size = sample_size, replace = True, p = p)
#     n_species = np.unique(species, return_counts = True)
    
#     sp_names = list(adj_sp_props.keys())
#     for sp, freq in zip(*n_species):
#         sp_name = sp_names[sp]
#         sp_data = data[data['Name']==sp_name]
        
#         if freq>len(sp_data):
#             n = len(sp_data)
#             print("Cannot take a larger sample ({}) than population ({}) whitout\
#                   replacement for species {}".format(freq, n, sp_name))
#             print('Drawing {} samples'.format(n))
#             print('\n')
#             sample = sp_data.sample(n, replace = False)
        
#         else:
#             sample = sp_data.sample(freq, replace = False)
            
#         adj_data = adj_data.append(sample)
        
#     return adj_data
    




