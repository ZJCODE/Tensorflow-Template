# -*- coding: utf-8 -*-
"""
Created on 2018.8.7

@author: zhangjun
"""


"""
Example :

python -B data_summary.py -data_path ../../data/ -data_name train_data

if use wordsvec data_name is words [run make_words in data_word2vec]

python -B data_summary.py -data_path ../../data_word2vec/ -data_name words


Output:

category_summary
numerical_summary
vocabulary_files

"""

import sys
sys.path.append("..")

import argparse
import config.config as config
import numpy as np
import json

head_dict = dict(zip(config.HEADER,range(len(config.HEADER))))

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', default='../../data/', type=str, help='data path')
parser.add_argument('-data_name', default='train_data', type=str, help='data path')
parser.add_argument('-data_sep', default=' ', type=str, help='data sep')
parser.add_argument('-sequence_sep', default=',', type=str, help='sequence sep')


args = parser.parse_args()

raw_data_path = args.data_path + args.data_name
data_sep = args.data_sep
sequence_sep = args.sequence_sep


numerical_columns = config.NUMERICAL_COLUMNS
category_columns = config.CATEGORY_COLUMNS
sequence_columns = config.SEQUENCE_COLUMNS


if len(config.SHARE_EMBEDDING_COLUMNS) > 0:
    category_columns = config.SHARE_EMBEDDING_COLUMNS
    head_dict = dict(zip(config.SHARE_EMBEDDING_COLUMNS,range(len(config.SHARE_EMBEDDING_COLUMNS))))


category_summary = {}
numerical_summary = {'mean':{},'std':{},'min':{},'max':{}}
sequence_summary = {}

for col in numerical_columns:
    for s in numerical_summary.keys():
        numerical_summary[s][col] = 0


# prepare numerical summary
cnt = 0
numerical_x_sum = []
numerical_x2_sum = []
numerical_x_min = []
numerical_x_max = []

for col in  numerical_columns:
    numerical_x_sum.append(0)
    numerical_x2_sum.append(0)
    numerical_x_min.append(10e10)
    numerical_x_max.append(-10e10)
#--------------------------------------------------------------


# prepare category columns
vocabulary_files = []
vocabulary_counters = []    
    
for col in category_columns:
    f = open(args.data_path + 'vocabulary_' + col,'w')
    vocabulary_files.append(f)
    vocabulary_counters.append({})
#--------------------------------------------------------------


# prepare seq columns
sequence_vocabulary_files = []
sequence_vocabulary_counters = []    
    
for col in sequence_columns:
    f = open(args.data_path + 'sequence_vocabulary_' + col,'w')
    sequence_vocabulary_files.append(f)
    sequence_vocabulary_counters.append({})
#--------------------------------------------------------------



with open(raw_data_path,'r') as f:
    for line in f.readlines():
        cnt += 1
        if cnt % 50000 == 0 :
            print("process line %d"%(cnt))
        line_split = line.strip().split(data_sep)
        

        # process category columns
        for i,col in enumerate(category_columns):
            if line_split[head_dict[col]] in vocabulary_counters[i]:
                vocabulary_counters[i][line_split[head_dict[col]]] += 1
            else:
                vocabulary_counters[i][line_split[head_dict[col]]] = 1
        

        # get numerical summary
        for i,col in enumerate(numerical_columns):
            val = float(line_split[head_dict[col]])
            numerical_x_sum[i] += val
            numerical_x2_sum[i] += val * val
            numerical_x_max[i] = max(numerical_x_max[i],val)
            numerical_x_min[i] = min(numerical_x_min[i],val)


        # get sequence summary

        for i,col in enumerate(sequence_columns):
            sequence = line_split[head_dict[col]]
            for seq_word in sequence.split(sequence_sep):
                if seq_word in sequence_vocabulary_counters[i]:
                    sequence_vocabulary_counters[i][seq_word] += 1
                else:
                    sequence_vocabulary_counters[i][seq_word] = 1
    
    print("-------------------------------------------------")
    


    # write out numerical summary 
    for i,col in enumerate(numerical_columns):
        numerical_summary['mean'][col] = numerical_x_sum[i] / cnt  # mean = sum(x_i) / n
        numerical_summary['std'][col] = np.sqrt(1.0/(cnt-1) * ( numerical_x2_sum[i] - cnt * np.square(numerical_summary['mean'][col])))  # std = sqrt(1/(n-1) * (sum(x_i^2) - n*mean^2))
        numerical_summary['max'][col] = numerical_x_max[i]
        numerical_summary['min'][col] = numerical_x_min[i]
        
    numerical_summary_json = json.dumps(numerical_summary)
    with open(args.data_path + 'numerical_summary','w') as f:
        f.write(numerical_summary_json)
        

    # write out vocabulary counter and category_summary
    for i,col in enumerate(category_columns):
        print('%s cnt : %d'%(col,len(vocabulary_counters[i])))
        category_summary[col] = len(vocabulary_counters[i])
        for id,count in vocabulary_counters[i].items():
            vocabulary_files[i].writelines(str(id) + ',' + str(count) + '\n')
        vocabulary_files[i].close()
    category_summary_json = json.dumps(category_summary)
    with open(args.data_path + 'category_summary','w') as f:
        f.write(category_summary_json)
        

    # write out sequnece vocabulary counter and sequence_summary
    for i,col in enumerate(sequence_columns):
        print('%s cnt : %d'%(col,len(sequence_vocabulary_counters[i])))
        sequence_summary[col] = len(sequence_vocabulary_counters[i])
        for id,count in sequence_vocabulary_counters[i].items():
            sequence_vocabulary_files[i].writelines(str(id) + ',' + str(count) + '\n')
        sequence_vocabulary_files[i].close()
    sequence_summary_json = json.dumps(sequence_summary)
    with open(args.data_path + 'sequence_summary','w') as f:
        f.write(sequence_summary_json)


        