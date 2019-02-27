# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
from utils.util import list_files

from utils.conf import Config
from utils.data import dataProcess

config = Config(base_dir='../conf')


def input_fn(data_path, num_epochs, mode, batch_size):
    def parse_csv(value):
        columns = tf.decode_csv(value, field_delim=config.FIELD_DELIM, record_defaults=config.RECORD_DEFAULTS)
        features = dict(zip(config.HEADER, columns))
        return features

    # prepare dataset
    data_file_list = list_files(data_path)
    dataset = tf.data.TextLineDataset(data_file_list)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_csv, batch_size=batch_size))

    # self.dataset = self.dataset.batch(self.batch_size)
    # self.dataset = self.dataset.map(parse_csv, num_parallel_calls=config.NUM_PARALLEL)

    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=3 * batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(1)
    raw_features = dataset.make_one_shot_iterator().get_next()

    # processs label
    label_col = config.get_data_prop('label_col')
    labels = None
    if len(label_col) == 0:
        pass
    elif len(label_col) == 1:
        labels = raw_features.pop(label_col[0])
    else:
        labels = {}
        for col in label_col:
            labels.update({col: raw_features.pop(col)})

    # preprocess feature
    data_process_params = {'sequence_cols': config.SEQUENCE_COLS, 'map_extend_features': config.MAP_EXTEND_FEATURE}
    data_process = dataProcess(data_process_params)
    raw_features = data_process.do_process(raw_features)

    return raw_features, labels


if __name__ == '__main__':
    with tf.Session() as sess:
        features, labels = input_fn(data_path=config.get_data_prop('train_data_path'),
                                    num_epochs=2,
                                    mode='eval',
                                    batch_size=2)

        from features import get_processed_feature

        feature_processed = get_processed_feature(features)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        a, b, c = sess.run([features, labels, feature_processed])
        # a = sess.run(features)

        for k, v in a.items():
            print(k)
            print(v.shape)
            print(v)
            print('\n')

        print(b)

        for k, v in c.items():
            print(k)
            print(v.shape)
            print(v)
            print('\n')
