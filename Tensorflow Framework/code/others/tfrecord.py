# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
import glob, multiprocessing
import shutil
import os


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    value = int(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    value = float(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = value.encode('utf8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _add_feature(value, data_type):
    if data_type == 'float':
        return _float_feature(value)
    elif data_type == 'int':
        return _int64_feature(value)
    elif data_type == 'str':
        return _bytes_feature(value)
    else:
        print('wrong data type')


def _prase_feature(data_type):
    if data_type == 'float':
        return tf.FixedLenFeature(shape=[], dtype=tf.float32)
    elif data_type == 'int':
        return tf.FixedLenFeature(shape=[], dtype=tf.int64)
    elif data_type == 'str':
        return tf.FixedLenFeature(shape=[], dtype=tf.string)
    else:
        print('wrong data type')


def instance2type(x):
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, float):
        return 'float'
    elif isinstance(x, int):
        return 'int'
    elif isinstance(x, str):
        return 'str'


class TFRecord:
    def __init__(self, raw_data_path, tfrecord_data_path, pool_num, params):
        self.raw_data_path = raw_data_path
        self.tfrecord_data_path = tfrecord_data_path
        self.raw_data_file_list = glob.glob(self.raw_data_path + '/*')
        self.pool_num = pool_num
        self.params = params  # sep , data_header ,data_type , output_file_max_length

    def write_tfrecord_one(self, file_name):
        file_num = 0
        cnt = 0
        filename = self.tfrecord_data_path + '/' + file_name.strip().split('/')[-1] + '_{}.tfrecords'.format(file_num)
        print('write tfreocrd : {}'.format(filename))
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(filename)
        with open(file_name, 'r') as f:
            for line in f:
                split_line = line.rstrip().split(self.params['sep'])
                # Create a feature
                feature = {}
                for index, (col_name, data_type) in enumerate(
                        zip(self.params['data_header'], self.params['data_type'])):
                    feature.update({col_name: _add_feature(split_line[index], data_type)})
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                # Serialize to string and write on the file
                writer.write(example.SerializeToString())
                cnt += 1
                if cnt % self.params['output_file_max_length'] == 0:
                    writer.close()
                    file_num += 1
                    filename = self.tfrecord_data_path + '/' + file_name.strip().split('/')[
                        -1] + '_{}.tfrecords'.format(file_num)
                    print('write tfreocrd : {}'.format(filename))
                    writer = tf.python_io.TFRecordWriter(filename)

    def write_tfrecord(self):
        shutil.rmtree(self.tfrecord_data_path, ignore_errors=True)
        os.mkdir(self.tfrecord_data_path)
        pool = multiprocessing.Pool(self.pool_num)
        pool.map(self.write_tfrecord_one, self.raw_data_file_list)
        pool.close()
        pool.join()

    def _parse_exmp(self, serial_exmp):
        feature_spec = {}
        for col_name, data_type in zip(self.params['data_header'], self.params['data_type']):
            feature_spec.update({col_name: _prase_feature(data_type)})
        features = tf.parse_single_example(serial_exmp, features=feature_spec)
        return features

    def read_tfrecord(self, tfrecord_data_path, batch_size):
        data_file_list = glob.glob(tfrecord_data_path + '/*')
        dataset = tf.data.TFRecordDataset(data_file_list, num_parallel_reads=10)
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=self._parse_exmp, batch_size=batch_size))

        # dataset = dataset.map(self._parse_exmp, num_parallel_calls=10)
        # dataset = dataset.batch(batch_size)
        return dataset


if __name__ == '__main__':
    # write  ==========================================================

    params = {'sep': '',
              'data_header': [],
              'data_type': [],
              'output_file_max_length': 3}

    TFR = TFRecord(raw_data_path='',
                   tfrecord_data_path='',
                   pool_num=4,
                   params=params)
    TFR.write_tfrecord()

    # read   ==========================================================

    dataset = TFR.read_tfrecord(tfrecord_data_path='', batch_size=2)
