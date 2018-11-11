# -*- coding: utf-8 -*-
"""
Created on 2018.8.8

@author: zhangjun
"""

import os
import tensorflow as tf
import numpy as np



#--------------- func for dataset start ----------------------------------------

def parse_csv_row(csv_row,head,record_defaults,sep='\001'):
    columns = tf.decode_csv(csv_row,field_delim = sep ,record_defaults=record_defaults)
    row = dict(zip(head, columns))
    return row


def dict_lookup(dt,key,default = np.array([0]).astype(np.int32)[0] ):
    try:
        return dt[key]
    except:
        return default

vectorize_dict_lookup = np.vectorize(dict_lookup)

def category_reindex(row,feature_name,vocabulary_dict):
    row[feature_name] = tf.cast(tf.reshape(tf.py_func(lambda x : vectorize_dict_lookup(vocabulary_dict,x),[row[feature_name]],[tf.int32]),[-1]),tf.int32)
    return row

def category_seq_reindex(row,feature_name,vocabulary_dict,batch_size):
    row[feature_name] = tf.cast(tf.reshape(tf.py_func(lambda x : vectorize_dict_lookup(vocabulary_dict,x),[row[feature_name]],[tf.int32]),[batch_size,-1]),tf.int32)
    return row

def transform(row,feature_name,way):
    if way == 'log':
        row[feature_name + '_' + way] = tf.log(tf.clip_by_value(row[feature_name],1e-5,10e10))
    elif way == 'square':
        row[feature_name + '_' + way] = tf.square(row[feature_name])
    elif way == 'sqrt':
        row[feature_name + '_' + way]= tf.sqrt(row[feature_name])
    else:
        pass
    return row


def norm_scale(row,feature_name,numerical_summary):
	row[feature_name + "_norm_scale"] = (row[feature_name] - numerical_summary['mean'][feature_name]) / numerical_summary['std'][feature_name]
	return row

def max_min_scale(row,feature_name,numerical_summary):
    row[feature_name + "_maxmin_scale"] = (row[feature_name] - numerical_summary['min'][feature_name]) / (numerical_summary['max'][feature_name] - numerical_summary['min'][feature_name])
    return row


def onehot(row,feature_name,size):
    row[feature_name+"_onehot"] = tf.one_hot(row[feature_name],size)
    return row


def hash_one_hot(row,feature_name,hash_size):
    row[feature_name] = tf.one_hot(tf.string_to_hash_bucket(tf.cast(row[feature_name],tf.string),hash_size),hash_size)
    return row

def cross_feature(row,feature_name_1,feature_name_2,hash_size):
    row["cross_" + feature_name_1  + "_" + feature_name_2] = tf.one_hot(tf.string_to_hash_bucket(tf.string_join([tf.cast(row[feature_name_1],tf.string),
                                                                                                      tf.cast(row[feature_name_2],tf.string)]),hash_size),2*hash_size)
    return row

def string2dense(row,feature_name,sep=','):
    row[feature_name] = tf.sparse_tensor_to_dense(tf.string_split(row[feature_name],sep),default_value = '')
    return row


def get_sequence_length(row,feature_name,sep=','):
    # do before batch
    row[feature_name+'_length'] = tf.size(tf.string_split([row[feature_name]],sep).values)
    return row

#--------------- func for dataset end ----------------------------------------


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    path_split = path.split("/")
    for i in range(len(path_split)):    
        try:
            os.mkdir('/'.join(path_split[:i+1]))
        except OSError:
            pass


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
  
