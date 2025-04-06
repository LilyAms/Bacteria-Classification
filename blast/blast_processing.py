#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:07:47 2021

@author: Lily Amsellem
"""
import os
from Bio import SearchIO
import subprocess
import shlex
from data_processing import read_fasta                


def execute_cline(cmd, **kwargs):                                                                                                 
    subprocess.call(shlex.split(cmd))
    

if __name__=='__main__':
    
    print(os.getcwd())

    os.chdir('../BLAST_data/ncbi_16S_rRNA_db')
    mock_data_file = '../../mock_data/mock_com_28042021.fas'
    records = read_fasta(mock_data_file)
    species = [(record.id, record.seq) for record in records]
    
    
    for sp in species:
        iD, seq = sp
        query_file = '../query_seq.fas'
        with open(query_file,'w') as q_file:
            q_file.write('>{}'.format(iD))
            q_file.write('\n')
            q_file.write(str(seq))
            q_file.write('\n')
        blast_xml = '../blast_results/blast_hits_{}.xml'.format(iD)

        
        # Run BLAST 
        print('------------------------')
        print('1. Running BLAST on query: ', iD)
        blast_args = {'query_file':shlex.quote(query_file), 'out_file':shlex.quote(blast_xml), 
                      'db':shlex.quote('16S_ribosomal_RNA'), 'max_target_seqs':100, 
                      'perc_identity': 90, 'outfmt':5}
        blast_cmd = "blastn -query {query_file} -out {out_file} -db {db} \
            -perc_identity {perc_identity} -max_target_seqs {max_target_seqs} -outfmt {outfmt}".format_map(blast_args)
        execute_cline(blast_cmd, **blast_args)
        
        # Extract hit ids
        ids_file = '../blast_hit_sequences/hit_seq_ids_{}.txt'.format(iD)
        blast_result = SearchIO.read(blast_xml, 'blast-xml')
        seq_ids = [hit.id for hit in blast_result.hits]

        with open(ids_file,'w') as f:
            f.write('\n'.join(seq_ids))
        
        # Fetch sequences
        print('2. Retrieving hit sequences with {}% identity ...'.format(blast_args['perc_identity']))
        out_file = '../blast_hit_sequences/similar_sequences_{}.fst'.format(iD) 
        db_cmd_args = {'dbtype': shlex.quote('nucl'), 'db': shlex.quote('16S_ribosomal_RNA'), 
                               'entry_batch': shlex.quote(ids_file), 
                               'out': shlex.quote(out_file), 'outfmt': shlex.quote('%f')}
        
        db_cmd = "blastdbcmd -db {db} -dbtype {dbtype} -entry_batch {entry_batch} -out {out} -outfmt {outfmt}".format_map(db_cmd_args)

        execute_cline(db_cmd, **db_cmd_args)
        print('Extracted {} similar sequences'.format(len(seq_ids)))
    

