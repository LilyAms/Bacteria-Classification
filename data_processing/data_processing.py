#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 18:44:46 2021

@author: Lily Amsellem
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

# --------------------------------------------------- BIOPYTHON SPECIFICS ---------------------------------------------------

def read_fasta(file_path):
    records = []
    with open(file_path,"r") as fasta_file:
        records = list(SeqIO.parse(fasta_file,"fasta"))
    return records

def get_nucleotide_counts(seq):
    nuc_counts = {}
    for nuc in "ATCG":
        nuc_counts[nuc] = seq.count(nuc)
    return nuc_counts


def df_from_records(records):
    ids = [record.id for record in records]
    sequences = [str(record.seq) for record in records]
    df = pd.DataFrame({"name":ids, "sequence": sequences})
    return df


# --------------------------------------------------- GENERAL ---------------------------------------------------

##### VEC2SEQ AND SEQ2VECTOR TRANSLATION

def seq2vec(seq, nuc2int = None):
    """" Vectorise a Sequence object/string as a numpy array. 
    
    Parameters
    ----------
    seq: Sequence Object from BioPython
    nuc2int: An ordinal mapping from nucleotides to integers. 
    If None is provided, default used is nuc2int = {"A":1,"C":2,"G":3, "T":4}
        
    Returns
    -------
    vec_seq: a numpy array of size len(seq) containing the index of the corresponding nucleotide in the DNA sequence
    such that nucleotides = {"A":0,"C":1,"G":2, "T":3}
    
    """
    
    if nuc2int is None:
        nuc2int = {"A":1,"C":2,"G":3, "T":4}
    
    vec_seq = np.zeros(len(seq))
    for char in "ACGT":
        char_idx = [i for i,ltr in enumerate(seq) if ltr==char]
        vec_seq[char_idx] = nuc2int[char]
    
    return vec_seq
        
def vec2seq(vec, int2nuc = None):
    """" Retrieve a DNA sequence Biopython Object from a 1D numpy array. 
    
    Parameters
    ----------
    vec: the numpy array to transform into a sequence
        
    Returns
    -------
    seq: the DNA sequence corresponding to the input vec
    
    """
    
    if int2nuc is None:
        int2nuc = {0:"", 1:"A", 2:"C", 3:"G", 4:"T"}
    v2s = np.vectorize(lambda x:int2nuc[x])
    seq = "".join(list(v2s(vec)))
    seq = Seq(seq)
    return seq

def translate2nuc(x, int2nuc = None):
    """Translate a vectorized DNA sequence into its character equivalent. 
    
    Parameters
    ----------
    x: a vectorised DNA sequence: a string or an array made of integers in range [0,5]
    
    Returns
    -------
    A string of characters in 'A,T,C,G' which is the translation of input x
    
    """
    
    if type(x)==str:
        if int2nuc is None:
            int2nuc = {'0':"", '1':"A", '2':"C", '3':"G", '4':"T"}
        return x.translate(str.maketrans(int2nuc))
    
    else:
        if int2nuc is None:
            int2nuc = {0:"", 1:"A", 2:"C", 3:"G", 4:"T"}
        v2s = np.vectorize(lambda x:int2nuc[x])
        seq = "".join(list(v2s(x)))
        return seq

def extract_kmers(seq,k):
    """Extract overlapping k-mers (i.e. sequences of k nucleotides) from a DNA sequence. 
    If seq = 'AATCGTC', k = 4, will return ['AATC', 'ATCG', 'TCGT', 'CGTC']

    Parameters
    ----------
    seq : DNA sequence (string or Seq object)
    k : k-mer length

    Returns
    -------
    k_mers : list of all overlapping k-mers in seq.

    """
    k_mers = [seq[idx:idx + k] for idx in range(len(seq) - k + 1)]
    return k_mers

##### READ FROM FILES AND LOAD DATA

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
        if file_path.endswith('.fas') or file_path.endswith('.fasta') or file_path.endswith('.fa'):
            dna_records = list(SeqIO.parse(file,"fasta"))
            sequences = [record.seq for record in dna_records]
            
        else:
            sequences = file.read().split()
            
    return sequences


def pad_sequence(sequences,max_len):
    """"Pad a list of sequences to the same length. 
    
    Parameters
    ----------
    sequences: a list of vectorised sequences (numpy arrays) 
    max_len: the maximum length to pad all the sequences at
    
    Returns
    -------
    An array of shape (nb_sequences, max_len) containing vectorised sequences of equal length, padded with zeros. 
    
    """
    
    padded_sequences = np.zeros((len(sequences),max_len))
    for idx,sequence in enumerate(sequences):
        padded_sequences[idx, :len(sequence)] = sequence
    return padded_sequences


def load_data_as_df(data_file, labels_file):
    """Load data into a dataframe from text files containing the DNA sequences and their labels
    
    Parameters
    ----------
    data_file : txt file containing the DNA sequences
    labels_file : txt file containing the species identifier for each sequence

    Returns
    -------
    data_df : a pandas dataframe containing 2 columns: "sequence" for the vectorised DNA sequences 
    and "species" for the species id. 

    """
    data_df = pd.read_csv(data_file,names = ['sequence'])
    labels_df = pd.read_csv(labels_file, names = ['species'])
    data_df = pd.concat([data_df, labels_df], axis = 1)
    return data_df

def load_data_as_array(data_file, labels_file, nuc2int = None):
    """Load DNA sequences from a text file into a numpy array. Nucleotides will be converted to integers, and sequences padded 
    to maximum length. 

    Parameters
    ----------
    seq_file : a text file which contiains DNA sequences. 
    labels_file: a text file containing the species identifier for each DNA sequence
    nuc2int: ordinal encoding for the DNA nucleotides. If None, default is: nuc2int = {"A":1,"T":2,"C":3, "G":4}

    Returns
    -------
    data: a numpy array of shape (nb_sequences, max_len) containing vectorised sequences of equal length, padded with zeros. 
    labels: a numpy array containing the labels for the loaded sequences
    
    """
    
    # Get sequences from FASTA file
    sequences = read_sequences(data_file)
    # Vectorise sequences
    vec_sequences = [seq2vec(sequence, nuc2int) for sequence in sequences]
    max_len = max([len(seq) for seq in vec_sequences])
    data = pad_sequence(vec_sequences, max_len)
    labels = np.loadtxt(labels_file)
    
    return data, labels

def create_str_df(seq_df):
    """Translate vectorised sequences from a dataframe to strings of characters in {A,T,C,G}. 
    The "sequence" column of the dataframe will be modified in place. 

    Parameters
    ----------
    seq_df: a pandas dataframe containing a "sequence" with sequences of integers. 

    Returns
    -------
    seq_df : the dataframe with its 'sequence' column modified : 
        it now contains the original DNA sequences as character sequences. 

    """
    f_int2nuc = lambda x:translate2nuc(x['sequence'])
    seq_df['sequence'] = seq_df.apply(f_int2nuc, axis = 1 )
        
    return seq_df