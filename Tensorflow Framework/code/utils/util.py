# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
import time
import os


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


def get_free_gpus(use_mem_limit=500):
    """
    获得当前gpu上空闲的gpu编号，空闲定义：内存使用小于use_mem_limit
    :param use_mem_limit: 内存使用限制
    :return:['0', '1','7']
    """
    cmd = "nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader"
    results = os.popen(cmd).readlines()
    gpu_num = 0
    free_gpu_num = []
    for line in results:
        # print('gpu:{}, result:{}'.format(gpu_num, line))
        used_mem = int(line.split(',')[2].split(' ')[1].strip())
        if used_mem < use_mem_limit:
            free_gpu_num.append(str(gpu_num))
        gpu_num += 1
    print('free gpus:{}'.format(free_gpu_num))
    return free_gpu_num