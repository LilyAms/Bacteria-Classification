#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:40:02 2021

@author: Lily Amsellem
"""

# Imports
import numpy as np
import time
import os
import copy
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0,'../data_processing/')

from data_processing import translate2nuc, seq2vec



def read_sequences(file_path):
    """Read a sequence file and load sequences into a list. A Fasta file or a regular csv/txt file can be processed. 
    In the case of a text file, sequences should be separated with '\n'
    

    Parameters
    ----------
    file_path : path to sequence file

    Returns
    -------
    records : list of Biopython records (if file format is Fasta), else a list of DNA strings. 

    """

    with open(file_path,"r") as file:
        if file_path.endswith(('.fas','.fasta')) or file_path.endswith(('.fa','.fst')):
            dna_records = list(SeqIO.parse(file,"fasta"))
            ids = [record.id for record in dna_records]
            sequences = [str(record.seq) for record in dna_records]
            return ids, sequences
            
        else:
            sequences = file.read().split()
            return sequences


def get_sub(nucl_to_sub):
    """Choose random substitutions for an array of nucleotides
    
    Parameters
    ----------
    nucl_to_sub: numpy array of ints in range [0,4] corresponding to nucleotides to substitute
        
    Returns
    -------
    subs: a numpy array of substituted nucleotides
    
    """
    
    # For each nucleotide, choose a random substitution (different from the nucleotide itself)
    choose_sub = lambda x:np.random.choice([i for i in range(1,5) if i!=x])
    
    # Vectorize the function
    choose_sub_vec = np.vectorize(choose_sub)
    
    # Apply to the array of nucleotides
    subs = choose_sub_vec(nucl_to_sub)
    return subs

def get_error_positions_vec(error_rates, sequence):
    """" Choose random error positions in a sequence. 
        
    Parameters
    ----------
    error_rates: The rates of occurrence for each mutation type
    sequence: the DNA sequence to be mutated. 
        
    Returns
    -------
    error_pos: a numpy array containing all the indices of the errors in the sequence. 
    
    """
    sum_rate = np.sum(error_rates)
    error_pos = np.random.choice(np.arange(len(sequence)),size = int(sum_rate*len(sequence)))
    return error_pos

def get_rdm_mutation(num_errors, error_rates):
    """" Choose a random mutation (insertion, deletion or substitution) according to the error rates 
    for a number num_errors of errors.  
    
    Parameters
    ----------
    num_errors: the number of random mutations to apply.
    error_rates: the probability of occurrence of each error type.
    
    Returns
    -------
    mutations: a numpy array of size num errors containing integers corresponding to each error type: 0 for insertion,
    1 for deletion, 2 for substitution
    
    """
    
    p_error = error_rates/np.sum(error_rates)
    mutations = np.random.choice(np.arange(3), size = num_errors, p = p_error)
    return mutations
    

def add_mutations(seq, error_rates):
    """"
    
    Parameters
    ----------
    vec_seq: a numpy array (vectorised DNA sequence)
    error_rates: a list of error rates for each possible mutation
    
    Returns
    -------
    vec_seq: a numpy array with mutations according to error_rates
    mutations_record: a numpy array keeping track of the sequence mutations, each line  of the form: 
        [mutation_type, position, old nucleotide, new_nucleotide]
    
    """
    
    record = []    
    vec_seq = np.copy(seq)
    # Select error positions
    error_idx = get_error_positions_vec(error_rates, vec_seq)
    
    # Assign error types 
    error_types = get_rdm_mutation(len(error_idx), error_rates)
    
 
    # SUBSTITUTIONS
    # Get sequence indices of the substitutions
    sub_idx = error_idx[error_types == 2]
    # Check that there are substitutions to perform
    if len(sub_idx)>0:
        new_nucl =  get_sub(vec_seq[sub_idx])
        # Save substitution record
        record = record + [[2, sub_idx[i], vec_seq[sub_idx[i]], new_nucl[i]] for i in range(len(sub_idx))]
        # Substitute nucleotides at those positions
        vec_seq[sub_idx] = new_nucl
    
    
    # DELETIONS
    del_idx = error_idx[error_types==1]
    if len(del_idx)>0:
        # Save deletion record
        record = record + [[1, del_idx[i], vec_seq[del_idx[i]], None] for i in range(len(del_idx))]
        # set to -1 temporarily, to be removed at the end
        vec_seq[del_idx] = -1

    # INSERTIONS 
    ins_idx = error_idx[error_types==0]
    if len(ins_idx)>0:
        ins_values = np.random.choice(np.arange(1,5), size = len(ins_idx))
        # Save insertion record
        record = record + [[0, ins_idx[i], None, ins_values[i]] for i in range(len(ins_values))]
        vec_seq  = np.insert(vec_seq,ins_idx,ins_values)
    
    # Finally remove -1 for deletions
    vec_seq = vec_seq[vec_seq!=-1]
    
    # Make record of all mutations 
    record = np.array(record)
    
    return vec_seq, record

def generate_mutation_record(mut_record,label, record_file):
    """Generate a record file containing the detail of the mutations applied to a DNA sequence when running 
    the mutation simulator. 
    
    Parameters
    ----------
    mut_record: the list of mutations obtained from mutate_sequence function, of the form
    [(mutation_type, nucleotide_position, former_nucleotide, new_nucleotide)]
    record_file: the path to the folder where the record of the mutations will be kept
    
    Returns
    -------
    None 
    
    """
    
    with open(record_file, 'a') as r_file:
        header = "MUT_TYPE, POS, FORMER_NUC, NEW_NUC"
        r_file.write('species: ' + str(label) + '\n')
        np.savetxt(r_file, mut_record, header = header, fmt = "%s")
        r_file.write('\n')


def mutate_sequences(ids, sequences, error_rates, species_distribution, sample_size, record_path = None):
    """ Mutate a list of DNA sequences. 
    For each input sequence, N_it sequences with random mutations will be generated. 
    
    Parameters
    ----------
    ids: a list of species names/ids corresponding to each DNA sequence
    sequences : list of DNA sequences (strings) of various lengths
    error_rates : list of error rates corresponding to each mutation type [insertion_rate, deletion_rate, substitution_rate]
    species_distribution: numpy array containing the number of mutated versions to be generated for each species (in order). 
    record_file: the path to the folder where mutation records should be saved. Default is None if no saving is required. 
    
    Returns
    -------
    mutated_sequences: a list of  length N_it containing vectorised mutated sequences 
    labels: a list of labels indicating the species label for each sequence

    """
    
    # Vectorise sequences
    vec_sequences = [seq2vec(sequence) for sequence in sequences]
    
    mutated_records = []
    
    # Prepare text files to save mutated sequences, labels and mutation report
    if record_path:
        new_dir = os.path.join(record_path,"in_{}_del_{}_sub_{}".format(*error_rates))
        try:
            os.mkdir(new_dir)
        except OSError:
            pass
        
        record_file = os.path.join(new_dir, "mutation_records.txt")
        mut_seq_file = os.path.join(new_dir, "sequences.fst")
        
        with open(record_file, 'a') as r_file:
            r_file.write("insertions,deletions,substitutions: ")
            r_file.write(str(error_rates))
            r_file.write('\n')
                  
    print("Generating mutations...")
    print()
    print("error rates - insertion {}, deletion {}, substitution {}".format(*error_rates))
    print()
    
    # Generate N_it different mutated sequences for each original DNA sequence
    start = time.time()
    
    if species_distribution is None:
        # If species_distribution is None, we generate equal proportions of each species
        species_distribution = [sample_size//len(ids) for i in range(len(ids))]
        
    for idx, (iD,seq) in enumerate(zip(ids,vec_sequences)):
        for i in range(int(species_distribution[idx])):
            mut_seq, mut_record = add_mutations(seq, error_rates)
            mutated_records.append(SeqRecord(Seq(translate2nuc(mut_seq)),id = iD))
            #if record_path:
                #generate_mutation_record(mut_record, iD, record_file)
                # Commented this out so no mutation records are generated anymore --> the txt files are too heavy
    
    if record_path:
        with open(mut_seq_file,'w') as fasta_file:
            SeqIO.write(mutated_records,fasta_file,format = 'fasta')

    
    print("Generated {} sequences in {}".format(len(mutated_records), time.time() - start))
    print()
    
    return mutated_records


def generate_mutation_data(seq_file, error_rates, species_distribution, sample_size, record_path = None):
    """
    Run the mutation simulator on a fasta file containing ATCG DNA sequences. 
    
    Parameters
    ----------
    seq_file : path to a file text or fasta) containing the original DNA sequences.
    error_rates : list of error rates corresponding to each mutation. 
    N_it : number of mutated versions of each sequence which are to be generated.
    record_path: the path to the folder where mutation records should be saved. Default is None if no saving is required. 

    Returns
    -------
    data : a numpy array containing the mutated sequences (N_it sequences for each sequence in seq_file)
    labels : a numpy array of labels for entries in the data array 

    """
    # Get sequences from FASTA file
    ids, sequences = read_sequences(seq_file)
    
    # Mutate sequences and get labels
    mutated_records = mutate_sequences(ids, sequences, error_rates, species_distribution, sample_size, record_path)

    return mutated_records

