# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
import time


def list_files(input_data):
    """if input file is a dir, convert to a file path list
    Return:
         file path list
    """
    if tf.gfile.IsDirectory(input_data):
        file_name = [f for f in tf.gfile.ListDirectory(input_data) if not f.startswith('.')]
        return [input_data + '/' + f for f in file_name]
    else:
        return [input_data]


def elapse_time(start_time):
    return round((time.time() - start_time) / 60)

def prase_feature(data_type):
    if data_type == 'float':
        return tf.FixedLenFeature(shape=[], dtype=tf.float32)
    elif data_type == 'int':
        return tf.FixedLenFeature(shape=[], dtype=tf.int64)
    elif data_type == 'string':
        return tf.FixedLenFeature(shape=[], dtype=tf.string)
    else:
        print('wrong data type')

def prase_feature_placehold(data_type):
    if data_type == 'float':
        return  tf.placeholder(tf.float32, [None])
    elif data_type == 'int':
        return  tf.placeholder(tf.int32, [None])
    elif data_type == 'string':
        return tf.placeholder(tf.string, [None])
    else:
        print('wrong data type')