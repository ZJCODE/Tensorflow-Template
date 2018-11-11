# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

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
