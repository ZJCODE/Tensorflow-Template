# -*- coding: utf-8 -*-
"""
Created on 2018.9.28

@author: zhangjun
"""

import tensorflow as tf
import config
from utils.util import list_files


class Dataset:
    def __init__(self, data_file, num_epochs, batch_size, shuffle):
        self.dataset = None
        self.data_file = data_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._read_dataset()
        self._preprocess()

    def _read_dataset(self):

        def parse_csv(value):
            columns = tf.decode_csv(value, field_delim=config.FIELD_DELIM, record_defaults=config.RECORD_DEFAULTS)
            features = dict(zip(config.HEADER, columns))
            return features

        data_file_list = list_files(self.data_file)

        print(data_file_list)

        self.dataset = tf.data.TextLineDataset(data_file_list)

        # self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
        #     map_func=parse_csv, batch_size=self.batch_size))

        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.map(parse_csv, num_parallel_calls=config.NUM_PARALLEL)

        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=3 * self.batch_size)
        self.dataset = self.dataset.repeat(self.num_epochs)
        self.dataset = self.dataset.prefetch(1)

    def _preprocess(self):

        def several_values_columns_to_array(row, feature_name):
            row[feature_name] = tf.sparse_tensor_to_dense(
                tf.string_split(row[feature_name], config.SEVERAL_VALUES_COLUMNS_SEP),
                default_value='')
            return row

        if len(config.SEVERAL_VALUES_COLUMNS) > 0:
            for key in config.SEVERAL_VALUES_COLUMNS:
                self.dataset = self.dataset.map(lambda row: several_values_columns_to_array(row, key),
                                                num_parallel_calls=config.NUM_PARALLEL)


def input_fn(data_file, num_epochs, mode, batch_size):
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
        num_epochs = 1

    Data = Dataset(data_file, num_epochs, batch_size, shuffle)
    features = Data.dataset.make_one_shot_iterator().get_next()
    labels = features.pop(config.LABEL_COLUMN)
    return features, labels


if __name__ == '__main__':
    with tf.Session() as sess:
        features, labels = input_fn(data_file=config.DATA_PATH + '/train',
                                    num_epochs=2,
                                    mode='eval',
                                    batch_size=2)

        b, c = sess.run([features, labels])

        print(b)
        print(c)
