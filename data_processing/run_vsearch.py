#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:59:24 2021

@author: Lily Amsellem
"""

    
import shlex
import subprocess
from os import path, listdir, makedirs
from data_selection import make_dataset
import pandas as pd
from data_generation import read_fasta
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord



columns = ['query', 'target', '% identity', 'alignment length',
           'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']

def execute_cline(cmd, **kwargs):                                                                                                 
    subprocess.call(shlex.split(cmd))
    


if __name__=='__main__':

    db_dir = '../../raw_data/mock_data/full_mock_com_28042021.fas'
    query_dir = '../../simulated_datasets/'
    query_exp_dir = '20210828_range_error_rates_17_species/even_distr/'
    
    
    query_file = path.join(query_dir, path.join(query_exp_dir, '_full_dataset.fas'))
    out_dir = path.join('../../vsearch/', query_exp_dir)
    out_file = path.join(out_dir, 'vsearch_hits_97perc_id.txt')
    
    # ------------------------- MOCK DATA SIMILARITY -----------------------
    # Run vsearch to find alignments between sequences in the mock data
    # out_dir = '../../raw_data/mock_data/'
    # records, ids, _, _ = read_fasta(db_dir)
    # species = [(record, seq) for (record,seq) in zip(records, ids)]
       
    # for sp in species:
    #     # create a query file containing the sequence in a fasta format for BLAST search
    #     iD, seq = sp
    #     query_file = '../../raw_data/mock_data/query_seq.fas'
    #     with open(query_file,'w') as q_file:
    #         q_file.write('>{}'.format(iD))
    #         q_file.write('\n')
    #         q_file.write(str(seq))
    #         q_file.write('\n')
    #     out_file = path.join(out_dir, 'vsearch_hits_{}.txt'.format(iD))
    
    # vsearch_args = {'query_file': shlex.quote(query_file), 'out_file': shlex.quote(out_file), 
    #           'db_dir':shlex.quote(db_dir),'perc_identity': 0.40, 'maxaccepts' : 0, 'maxrejects':0}
    
    # #Setting maxaccepts and maxrejects to 0 to get all hits and not only the top 1
    # vsearch_cmd = "vsearch --usearch_global {query_file} --db {db_dir} --id {perc_identity}\
    #     --blast6out {out_file} --maxaccepts {maxaccepts} --maxrejects {maxrejects} ".format_map(vsearch_args)
    # execute_cline(vsearch_cmd, **vsearch_args)

    
    #  ---------------------- Group all sequences into a single fasta file ---------------------------
    data_dir = path.join(query_dir, query_exp_dir)
    record_paths = sorted([f for f in listdir(data_dir) if not f.startswith('.')
                          and (f.startswith('nb_it') or f.startswith('in_'))])
    
    record_paths = [path.join(data_dir, path.join(rec_path,"sequences.fst"))  for rec_path in record_paths]
    
    data = make_dataset(record_paths)
    
    bio_records = []
    
    for idx, line in data.iterrows():
        iD, seq = str(idx), line['Sequence']
        bio_records.append(SeqRecord(Seq(seq),id = iD))
        
    with open(query_file,'w') as fasta_file:
        SeqIO.write(bio_records,fasta_file, format = 'fasta')
    
    # ----------------------- Run vsearch -----------------------------------------
    
    vsearch_args = {'query_file': shlex.quote(query_file), 'out_file': shlex.quote(out_file), 
                  'db_dir':shlex.quote(db_dir),'perc_identity': 0.40}
    
    vsearch_cmd = "vsearch --usearch_global {query_file} --db {db_dir} --id {perc_identity}\
        --blast6out {out_file}".format_map(vsearch_args)
        #--alnout
    
    try:
        makedirs(out_dir)
    except OSError:
        pass
        
    execute_cline(vsearch_cmd, **vsearch_args)
    
        
    # # ---------------------- Process output of vsearch --------------------------------
    
    # Read vsearch output
    vsearch_tab = pd.read_csv(out_file, sep = '\t', names = columns)
    ref_data = data.reset_index()
    # ref_data = parse_fasta(query_file)
    # ref_data = ref_data.reset_index()
    
    hits = ref_data.merge(vsearch_tab, left_on = 'index', right_on = 'query')
    hits = hits.drop('index', axis = 1)
    hits = hits[hits.Name == hits.target]
    
    rec_err_rates = hits['Error Rate'].unique()
    unrec_species = ref_data[ref_data['Error Rate'].isin(rec_err_rates) & -(ref_data['index'].isin(hits['query']))]
    
    unrec_species_grp = unrec_species.groupby(['Error Rate', 'Name']).count()
    
    #grp_dir = 'even_distr_unrec_species.csv'
    #grp_dir = path.join(out_dir, grp_dir)
    #unrec_species_grp.to_csv(grp_dir)
    
    
    
    
    