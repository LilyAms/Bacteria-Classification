#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:49:20 2021

@author: Lily Amsellem
"""

from data_generation import parse_fasta, write_df_to_fasta
import re

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


def find_species_names(data, species_ids):
    
    target_species = data[data['Name'].isin(species_ids)]

    target_species['Species'] = target_species.Description.apply(lambda x:x[-1])

    full_labels = target_species.loc[:,['Name','Species']]

    return full_labels  

def process_string(s):
    
    # decapitalize
    s_new = s.lower()
    
    # tokenize and remove punctuation
    re_punctuation_string = '[\s,\_,\/.\']'
    s_new = re.sub(re_punctuation_string, ' ', s_new)
    
    # remove non-character words
    s_new = re.sub(r'[0-9]+', '', s_new)
    s_new = re.sub(' sp', '', s_new)
    
    s_split = re.split(' ', s_new)

    return ' '.join(s_split[:2])

def process_list(l_strings):
    
    res = []
    
    for s in l_strings:
        s_new = process_string(s)
        res.append(s_new)
    
    return res



def read_dataset(file):
    """
    Create a Pandas DataFrame from a Fasta File containing biological sequences.
    Clean dataset from unknown nucleotides, replace protein sequences by corresponding DNA sequences.
    Add Description and Species name, and process species name using regexp for easier grouping and classification.  

    Parameters
    ----------
    file : path to fasta file containing the data.
   
    Returns
    -------
    new_df : DataFrame containing the following columns derived from the Fasta file: Name, Sequence, Description, regexp_sp_label

    """
    
    df = parse_fasta(file)
    print("Parsed Fasta file, reading {} sequences".format(len(df)))
    print('\n')
    
    
    # If protein sequences, change Us back to Ts to obtain DNA sequences
    df['Sequence'] = df['Sequence'].apply(lambda x:x.replace('U','T'))
    
    # Remove unique identifier from Description for convenience
    df['Description'] = df.apply(lambda x:x['Description'].replace(x['Name'] + ' ',''), axis = 1)
    df['Description'] = df['Description'].apply(lambda x:x.split(';'))  
    df['Species'] = df.Description.apply(lambda x:x[-1])
    
    
    # If the sequences contain 'n's, e.g. unknown nucleotides, remove those sequences
    contains_unknowns = df[df['Sequence'].str.contains('n')]
    df.drop(contains_unknowns.index, inplace = True)
    df.reset_index(inplace = True, drop = True)
    print('Removed {} sequences because of unknown nucleotides'.format(len(contains_unknowns)))
    print('\n')
    
    df['regexp_sp_label'] = df['Species'].apply(lambda x: process_string(x))
    ref_species = process_list(list(zymo_species.keys()))
    find_ref_species = df['regexp_sp_label'].str.contains('|'.join(sp for sp in ref_species))
    new_df = df[find_ref_species]
    
    print("Kept {} sequences out of {} sequences".format(len(new_df), len(df)))
    print('\n')
    
    return new_df
    
def sample_dataset(df, max_sample_size):
    """
    From species in df, if they appear several times in the dataframe,
    draw max_sample_size samples from each species. If a species appears less than max_sample_size, we draw all samples from
    that species.

    Parameters
    ----------
    df : pandas dataframe
        dataframe with the columns 'Sequence', 'Name', and 'regexp_sp_label'.
    max_sample_size : int
        number of samples to draw from each species.

    Returns
    -------
    first_sample : pandas dataframe
        dataframe with each species in the intial df sampled max_sample_size times.
    """
    
    # Removing duplicate sequences
    duplicates = df.duplicated(subset = ['Sequence'])
    new_df = df[~duplicates]
    
    # Randomly sample frac of the initial dataset
    sp_grp = new_df.groupby('regexp_sp_label').count()

    grp1_species = sp_grp[sp_grp['Name']>=max_sample_size].index.tolist()
    grp1 = new_df['regexp_sp_label'].str.contains('|'.join(sp for sp in grp1_species))
    first_sample = new_df[grp1].groupby('regexp_sp_label').sample(n = max_sample_size, random_state = 42).reset_index(drop = True)
    
    grp2_species = sp_grp[sp_grp['Name']<max_sample_size].index.tolist()

    for sp in grp2_species:
        sp_sample = new_df[new_df['regexp_sp_label'].str.contains(sp)].sample(n = sp_grp.loc[sp,'Name'], random_state = 42).reset_index(drop = True)
        first_sample = first_sample.append(sp_sample, ignore_index = True)

    first_sample['Description'] = first_sample['Description'].apply(lambda x: ';'.join(x))
    return first_sample
   

        

if __name__=='__main__':
    
    # Once species are sampled as required, write the df as a fasta file
    # Will serve as the input fasta file to the mutation simulator
    
    # Original database with all the 16S sequences
    dataset_file = '../../raw_data/large_mock_data/SILVA_138.1_SSURef_NR99_tax_silva.fasta'
    
    # Read and filter dataset
    seq_df = read_dataset(dataset_file)
    max_sample_size = 6
    # Sample 6 strains from each species
    sample_df = sample_dataset(seq_df, max_sample_size = max_sample_size)
    
    # Path to output fasta file
    out_file = '../../raw_data/new_mock_data/20210829_sample_{}_species.fas'.format(len(sample_df))

    # Write output fasta file
    write_df_to_fasta(sample_df, out_file)
