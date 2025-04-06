#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:12:22 2021

@author: Lily Amsellem
"""

from os import listdir, path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from data_processing import translate2nuc

# Aim : generate a dataset from multiple fasta files

# STEPS: 
    # 1) Read FASTA files
    # 2) Take all the sequences and re-write them into a full file ? 
    # 2 bis) Remove duplicates
    # 3) Binarize the labels 
    # 4) Make a dataframe out of this
    
def read_fasta(file_path, format = 'fasta'):
    records = list(SeqIO.parse(file_path, format = format))
    ids = [record.id for record in records]
    sequences = [str(record.seq) for record in records]
    desc = [record.description for record in records]
    return records, ids, sequences, desc
    
def parse_fasta(file_path, format = 'fasta', blast = False):
    records, ids, sequences, desc = read_fasta(file_path, format = format)
    
    seq_df = pd.DataFrame({'Name': ids, 'Sequence': sequences, 'Description':desc})
    
    # Remove unneeded information from Description column
    if blast and len(desc)>0:
        seq_df['Description'] = desc
        seq_df['Description'] = seq_df['Description'].apply(lambda x:x.replace(
            '16S ribosomal RNA, partial sequence','').replace('16S ribosomal RNA, complete sequence',''))
    
        # Add a reference to the query related to those sequences in BLAST
        blast_query = path.basename(path.normpath(file_path))
        seq_df['Related_blast_query'] = blast_query.replace('similar_sequences_','').replace('.fst','')
        
    return seq_df


def write_df_to_fasta(df, out_file):
    """
    Write a pandas DataFrame to a file in fasta format. 

    Parameters
    ----------
    df : DataFrame containing the columns Name, Sequence, and Description at least.
    out_file : str, path to the directory where the fasta file should be saved.

    Returns
    -------
    None.

    """
    
    df['SeqRecord'] = df.apply(lambda x:SeqRecord(Seq(x['Sequence']), id = x['Name'], 
                                                          description = x['Description']), axis = 1)
    
    records = df['SeqRecord'].tolist()
    
    if out_file is not None:
        with open(out_file,'w') as fasta_file:
            SeqIO.write(records,fasta_file,format = 'fasta')
    
    
if __name__=='__main__':
    
    # AIM: READ SEQUENCES FROM BLAST FASTAS
    # GROUP THEM INTO A SINGLE FASTA
    # MAKE A CSV RECORD BY PUTTING TEM INTO A DATAFRAME AND WRITING THEM INTO A CSV
    blast_loc_dir = '../BLAST_data/blast_hit_sequences/BLAST_alignments/'
    blast_files = sorted([f for f in listdir(blast_loc_dir)  if not f.startswith('.')
                          and f.startswith('similar_sequences')])
    
    full_records = []
    df = pd.DataFrame(columns = ['Name', 'Sequence'])
    
    for file in blast_files:
        # Keep records to write them to a FASTA
        records = read_fasta(blast_loc_dir + file)
        full_records = full_records + records[0]
        new_rows = parse_fasta(blast_loc_dir + file, blast = True)
        df = df.append(new_rows, ignore_index = True)
    
    
    # ----------------------- Extract 2 similar species per query ----------------------
    # Sample 2 rows from each group
    records_by2 = []
    select_sp = df.groupby('Related_blast_query')[['Name']].sample(
        n = 2,replace = False,random_state = 0)
    select_sp['Sequence'] = select_sp['Sequence'].apply(lambda x:Seq(x))
    
    # Convert the extracted samples to records
    for idx, row in select_sp.iterrows():
        records_by2.append(SeqRecord(row['Sequence'], id = row['Name']))
    
    # Write the records to a fasta file
    # out_fasta_file = 'BLAST_two_seq_per_query.fst'
    # out_csv_file = 'BLAST_two_seq_per_query.csv'
    # SeqIO.write(records_by2, '../BLAST_data/blast_hit_sequences/' + out_fasta_file,'fasta')
    
    # select_sp.to_csv('../BLAST_data/blast_hit_sequences/' + out_csv_file, index = False)
    
    # ------------------ Write all records to FASTA file and csv ---------------------
    #out_fasta_file = 'BLAST_all_sequences.fst'
    #SeqIO.write(full_records, blast_loc_dir + out_fasta_file, 'fasta')
    
    # Write details and data to dataframe
    # seq_df.groupby('Related_blast_query')[['Name']].count()
    #out_csv_file = 'BLAST_all_sequences.csv'
    #seq_df.to_csv(blast_loc_dir + out_csv_file, index = False)
    # Retrieve seq_df 
    # blast_df = pd.read_csv(out_csv_file)
    # new_sp_counts = blast_df.groupby('Related_blast_query')[['Name']].count()
    
    
    #mutations_loc_dir = '../simulated_data/eq_prop_mutations/'
    #mutated_files = sorted([f for f in listdir(mutated_loc_dir)  if not f.startswith('.') ])
    

    
   # for file in mutated_files:
        #new_rows = parse_fasta(mutations_loc_dir + file, blast = False)
        #seq_df = seq_df.append(new_rows, ignore_index = True)
    

        
        
        