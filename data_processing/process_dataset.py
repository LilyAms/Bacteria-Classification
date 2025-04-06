#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:49:20 2021

@author: Lily Amsellem
"""

from data_generation import parse_fasta, write_df_to_fasta
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import re

original_species = ['Akkermansia_muciniphila',
        'Bacteroides_fragilis',
        'Bifidobacterium_adolescentis',
        'Clostridioides_difficile',
        'Clostridium_perfringens',
        'Enterococcus_faecalis', 
        'Escherichia_coli',
        'Faecalibacterium_prausnitzii',
        'Fusobacterium_nucleatum',
        'Lactobacillus_fermentum',
        'Methanobrevibacter_smithii',
        'Prevotella_corporis',
        'Roseburia_hominis',
        'Salmonella_enterica',
        'Veillonella_rogosae',
        'Bacteroides_caccae',
       'Bacteroides_dorei', 
       'Bacteroides_ovatus', 
       'Bacteroides_plebeius',
       'Bacteroides_vulgatus', 
       'Bifidobacterium_bifidum',
       'Bifidobacterium_breve', 
       'Bifidobacterium_catenulatum',
       'Bifidobacterium_longum', 
       'Enterococcus_caccae',
       'Enterococcus_faecium', 
       'Parabacteroides_distasonis',
       'Parabacteroides_johnsonii', 
       'Parabacteroides_merdae',
       'Roseburia_faecis', 
       'Roseburia_intestinalis']

def read_dataset(file):
    """
    Create a Pandas DataFrame from a Fasta File containing biological sequences.
    Clean dataset from unknown nucleotides, replace protein sequences by corresponding DNA sequences.

    Parameters
    ----------
    file : path to fasta file containing the data.
    sample_size: Number of sequences to sample from dataset. By default, the full dataset will be returned. 
    U_or_T : Whether sequences should contain Us (for a protein sequence) or T (corresponding DNA sequence). 
    The default is 'T'.
    keep_ns : Whether or not to keep sequences wit unknown nucleotides in the database. 
    The default is False, sequences with ns will be removed.
    include_species : list of strings containing the species to include in the dataframe
   
    Returns
    -------
    bacteria_df : DataFrame containing the following columns derived from the Fasta file: Name, Sequence, Description.

    """
    
    df = parse_fasta(file)
    print("Parsed Fasta file, reading {} sequences".format(len(df)))
    print('\n')
    
    
    # If protein sequences, change Us back to Ts to obtain DNA sequences
    df['Sequence'] = df['Sequence'].apply(lambda x:x.replace('U','T'))
    
    # Remove unique identifier from Description for convenience
    df['Description'] = df.apply(lambda x:x['Description'].replace(x['Name'] + ' ',''), axis = 1)
    df['Description'] = df['Description'].apply(lambda x:x.split(';'))  
    
    # If the sequences contain 'n's, e.g. unknown nucleotides, remove those sequences
    contains_unknowns = df[df['Sequence'].str.contains('n')]
    df.drop(contains_unknowns.index, inplace = True)
    df.reset_index(inplace = True, drop = True)
    print('Removed {} sequences because of unknown nucleotides'.format(len(contains_unknowns)))
    print('\n')
        
    print("Filtering dataset to remove non-bacterial species...")
    print('\n')
    bacteria = df['Description'].apply(lambda x: 'Bacteria' in x)
    bacteria_df = df[bacteria]
    
    print("Kept {} bacterial sequences out of {} sequences".format(len(bacteria_df), len(df)))
    print('\n')
    
    return bacteria_df
    
def sample_dataset(df, sample_size, include_species = None):
    
    regex_sp = []
    for sp in include_species:
        sp = re.sub('\_', ' ', sp)
        #contains_id = bool(re.search(r'\d', sp))
        # if contains_id:
        #     sp = re.sub(r'\b[a-zA-Z]{2,}\b','',sp)
        #     sp = re.sub(r'[ \t]','',sp)
        regex_sp.append(sp)
    
    if include_species is not None:
        df['Species'] = df.Description.apply(lambda x:x[-1])
        df['Genus'] = df.Description.apply(lambda x:x[-2])
        #add_species = df.Species.str.contains('|'.join(regex_sp))
        add_species = df.Species.apply(lambda x:any(x==s for s in regex_sp)) # 6799
        
        hit_species = df[add_species].Species.unique().tolist()
        missing_sp = [sp for sp in regex_sp if sp not in hit_species]
        add_missing_species = df.Species.apply(lambda x:any(s in x for s in missing_sp))

        # Related genus
        related_genus = df[add_species | add_missing_species].Genus.unique().tolist()
        # Randomly sample frac of the initial dataset
        unvalid = df.Species.str.contains('uncultured|unidentified|metagenome')
        same_genus = df.Genus.isin(related_genus) & ~unvalid
        
        ref_sample = df[add_species | add_missing_species].groupby('Species').sample(random_state = 42).reset_index(drop = True) # 30
        selected_samples = ref_sample.Name.tolist()    
        selected_sp = ref_sample.Species.tolist()
        
        sample = df[~df.Name.isin(selected_samples) & ~df.Species.isin(selected_sp) & same_genus].groupby('Species').sample(n = 1, replace = False, random_state = 42).reset_index(drop = True)
        final_sample = sample.sample(n = sample_size - len(ref_sample), replace = False, random_state = 42).reset_index(drop = True) 
        final_sample = final_sample.append(ref_sample, ignore_index = True)
    
    else:
        final_sample = df.sample(n = sample_size, replace = False, random_state = 42).reset_index(drop = True)
    
    final_sample['Description'] = final_sample['Description'].apply(lambda x: ';'.join(x))
    return final_sample
   

        

if __name__=='__main__':
    
    dataset_file = '../../raw_data/large_mock_data/SILVA_138.1_SSURef_NR99_tax_silva.fasta'
    # MODIFY HERE:
    # Fraction of the dataset to be sampled (original dataset size: 510508)
    sample_sizes = [200,300,400,500]
    
    # Read and filter dataset
    seq_df = read_dataset(dataset_file)
    for sample_size in sample_sizes:
        sample_df = sample_dataset(seq_df, sample_size = sample_size, include_species = original_species) 
    
        # Path to output fasta file
        out_file = '../../raw_data/large_mock_data/created_exp_communities/20210804_sample_{}_species.fas'.format(len(sample_df))

        #Write output fasta file
        write_df_to_fasta(sample_df, out_file)
