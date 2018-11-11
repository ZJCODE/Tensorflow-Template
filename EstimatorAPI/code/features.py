# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

@author: zhangjun
"""

import tensorflow as tf
import numpy as np
import config
from datasets import input_fn

# notice later those need to change
# reference : https://www.tensorflow.org/api_docs/python/tf/contrib/feature_column
from tensorflow.contrib.lookup import HashTable
from tensorflow.contrib.lookup import KeyValueTensorInitializer
from tensorflow.contrib.feature_column import sequence_categorical_column_with_hash_bucket
from tensorflow.contrib.feature_column import sequence_input_layer
from tensorflow.contrib.feature_column import sequence_categorical_column_with_vocabulary_file


class Features:
    def __init__(self):

        # TODO Add Interact Feature

        self.hash_table = {}  # for feature extend

        self.numerical_feature_column = []

        self.bucketized_feature_column = []

        # one-hot
        self.categorical_feature_column_with_hash = {}
        self.categorical_feature_column_with_vocabulary_file = {}

        # embedding
        self.embedding_feature_column_with_hash = {}
        self.embedding_feature_column_with_vocabulary_file = {}

        # embedding separate feature
        self.embedding_feature_column_with_hash_separate = {}
        self.embedding_feature_column_with_vocabulary_file_separate = {}

        # one-hot matrix
        self.sequence_categorical_feature_column_with_hash = {}
        self.sequence_categorical_feature_column_with_vocabulary_file = {}

        # embedding matrix
        self.sequence_embedding_feature_column_with_hash = {}
        self.sequence_embedding_feature_column_with_vocabulary_file = {}

        # multi-hot
        self.multi_categorical_feature_column_with_hash = {}
        self.multi_categorical_feature_column_with_vocabulary_file = {}

        self._build()

    @staticmethod
    def _embedding_dim(dim):
        """empirical embedding dim"""
        return int(np.power(2, np.ceil(np.log(dim ** 0.25))))

    # ------------------------ feature_columns ------------------------

    # numerical
    def _get_numerical_feature_column(self):
        for key in config.NUMERICAL_COLUMNS:
            feature_col = tf.feature_column.numeric_column(key)
            self.numerical_feature_column.append(feature_col)

    # bucket
    def _get_bucketized_feature_column(self):
        for key, boundaries in config.BUCKET_COLUMNS:
            feature_col = tf.feature_column.numeric_column(key)
            feature_col = tf.feature_column.bucketized_column(feature_col, boundaries)
            self.bucketized_feature_column.append(feature_col)

    # ont-hot hash
    def _get_categorical_feature_column_with_hash(self):
        for key, hash_bucket_size in config.HASH_CATEGORICAL_COLUMNS:
            feature_col = tf.feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.categorical_feature_column_with_hash.update({key: feature_col})

    # ont-hot file
    def _get_categorical_feature_column_with_vocabulary_file(self):
        for key, file_path in config.FILE_CATEGORICAL_COLUMNS:
            file_len = len(open(file_path, 'r', encoding='utf8').readlines())
            feature_col = tf.feature_column.categorical_column_with_vocabulary_file(key, file_path, file_len)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.categorical_feature_column_with_vocabulary_file.update({key: feature_col})

    # embedding hash
    def _get_embedding_feature_column_with_hash(self):
        for key, hash_bucket_size, dimension in config.HASH_EMBEDDING_COLUMNS:
            if dimension == -1:  # dim_auto
                dimension = self._embedding_dim(hash_bucket_size)
            feature_col = tf.feature_column.categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string)
            feature_col = tf.feature_column.embedding_column(feature_col, dimension)
            self.embedding_feature_column_with_hash.update({key: feature_col})

    # embedding file
    def _get_embedding_feature_column_with_vocabulary_file(self):
        for key, file_path, dimension in config.FILE_EMBEDDING_COLUMNS:
            file_len = len(open(file_path, 'r', encoding='utf8').readlines())
            if dimension == -1:  # dim_auto
                dimension = self._embedding_dim(file_len)
            feature_col = tf.feature_column.categorical_column_with_vocabulary_file(key, file_path, file_len)
            feature_col = tf.feature_column.embedding_column(feature_col, dimension)
            self.embedding_feature_column_with_vocabulary_file.update({key: feature_col})

    # sequence ont-hot hash
    def _get_sequence_categorical_feature_column_with_hash(self):
        for key, hash_bucket_size in config.HASH_SEQUENCE_CATEGORICAL_COLUMNS:
            feature_col = sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.sequence_categorical_feature_column_with_hash.update({key: feature_col})

    # sequence ont-hot file
    def _get_sequence_categorical_feature_column_with_vocabulary_file(self):
        for key, file_path in config.FILE_SEQUENCE_CATEGORICAL_COLUMNS:
            file_len = len(open(file_path, 'r', encoding='utf8').readlines())
            feature_col = sequence_categorical_column_with_vocabulary_file(key, file_path, file_len)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.sequence_categorical_feature_column_with_vocabulary_file.update({key: feature_col})

    # sequence embedding hash
    def _get_sequence_embedding_feature_column_with_hash(self):
        for key, hash_bucket_size, dimension in config.HASH_SEQUENCE_EMBEDDING_COLUMNS:
            if dimension == -1:  # dim_auto
                dimension = self._embedding_dim(hash_bucket_size)
            feature_col = sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string)
            feature_col = tf.feature_column.embedding_column(feature_col, dimension)
            self.sequence_embedding_feature_column_with_hash.update({key: feature_col})

    # sequence embedding file
    def _get_sequence_embedding_feature_column_with_vocabulary_file(self):
        for key, file_path, dimension in config.FILE_SEQUENCE_EMBEDDING_COLUMNS:
            file_len = len(open(file_path, 'r', encoding='utf8').readlines())
            if dimension == -1:  # dim_auto
                dimension = self._embedding_dim(file_len)
            feature_col = sequence_categorical_column_with_vocabulary_file(key, file_path, file_len)
            feature_col = tf.feature_column.embedding_column(feature_col, dimension)
            self.sequence_embedding_feature_column_with_vocabulary_file.update({key: feature_col})

    # multi-hot hash
    def _get_multi_categorical_feature_column_with_hash(self):
        for key, hash_bucket_size in config.HASH_MULTI_CATEGORICAL_COLUMNS:
            feature_col = sequence_categorical_column_with_hash_bucket(key, hash_bucket_size, dtype=tf.string)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.multi_categorical_feature_column_with_hash.update({key: feature_col})

    # multi-hot file
    def _get_multi_categorical_feature_column_with_vocabulary_file(self):
        for key, file_path in config.FILE_MULTI_CATEGORICAL_COLUMNS:
            file_len = len(open(file_path, 'r', encoding='utf8').readlines())
            feature_col = sequence_categorical_column_with_vocabulary_file(key, file_path, file_len)
            feature_col = tf.feature_column.indicator_column(feature_col)
            self.multi_categorical_feature_column_with_vocabulary_file.update({key: feature_col})

    @staticmethod
    def _get_feature_columns_input(raw_feature, feature_col_name, feature_col):
        return {feature_col_name: tf.feature_column.input_layer(raw_feature, feature_col)}

    @staticmethod
    def _get_sequence_feature_columns_input(raw_feature, feature_col_name, feature_col):
        return {feature_col_name: sequence_input_layer(raw_feature, feature_col)[0]}

    @staticmethod
    def _get_multi_hot_feature_columns_input(raw_feature, feature_col_name, feature_col):
        return {feature_col_name: tf.reduce_sum(sequence_input_layer(raw_feature, feature_col)[0], axis=1)}

    # ------------------------ get feature and label ------------------------

    def get_processed_feature(self, raw_features):
        # raw_features the same sa result returned by input_fn [see the type in input_fn ]

        # ============================ preprocess raw features start ============================

        if len(config.MAP_EXTEND_FEATURE) > 0:
            for key, map_file, map_cols in config.MAP_EXTEND_FEATURE:
                self._build_hash_table(map_file, map_cols, config.EXTEND_FEATURE_SEP)
                raw_features = self._extend_more_feature(raw_features, key, map_cols)

        # ============================ preprocess raw features end ============================

        features = {}

        if len(config.NUMERICAL_COLUMNS) > 0:
            features.update(
                self._get_feature_columns_input(raw_features,
                                                'numerical_columns',
                                                self.numerical_feature_column))

        if len(config.BUCKET_COLUMNS) > 0:
            features.update(
                self._get_feature_columns_input(raw_features,
                                                'bucketized_column',
                                                self.bucketized_feature_column))
        # one-hot hash
        if len(config.HASH_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.HASH_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_feature_columns_input(raw_features,
                                                    'one_hot_column_with_hash_' + key,
                                                    # categorical_column_with_hash_
                                                    self.categorical_feature_column_with_hash[key]))
        # one-hot file
        if len(config.FILE_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.FILE_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_feature_columns_input(raw_features,
                                                    'one_hot_column_with_vocabulary_file_' + key,
                                                    # categorical_column_with_vocabulary_file_
                                                    self.categorical_feature_column_with_vocabulary_file[key]))

        # embedding hash
        if len(config.HASH_EMBEDDING_COLUMNS) > 0:
            for key, _, _ in config.HASH_EMBEDDING_COLUMNS:
                features.update(
                    self._get_feature_columns_input(raw_features,
                                                    'embedding_column_with_hash_' + key,
                                                    self.embedding_feature_column_with_hash[key]))
        # embedding file
        if len(config.FILE_EMBEDDING_COLUMNS) > 0:
            for key, _, _ in config.FILE_EMBEDDING_COLUMNS:
                features.update(
                    self._get_feature_columns_input(raw_features,
                                                    'embedding_column_with_vocabulary_file_' + key,
                                                    self.embedding_feature_column_with_vocabulary_file[key]))

        # sequence ont-hot hash
        if len(config.HASH_SEQUENCE_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.HASH_SEQUENCE_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_sequence_feature_columns_input(raw_features,
                                                             'sequence_one_hot_column_with_hash_' + key,
                                                             # sequence_categorical_column_with_hash_
                                                             self.sequence_categorical_feature_column_with_hash[key]))

        # sequence ont-hot file
        if len(config.FILE_SEQUENCE_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.FILE_SEQUENCE_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_sequence_feature_columns_input(raw_features,
                                                             'sequence_one_hot_column_with_vocabulary_file_' + key,
                                                             # sequence_categorical_column_with_vocabulary_file_
                                                             self.sequence_categorical_feature_column_with_vocabulary_file[
                                                                 key]))
        # sequence embedding hash
        if len(config.HASH_SEQUENCE_EMBEDDING_COLUMNS) > 0:
            for key, _, _ in config.HASH_SEQUENCE_EMBEDDING_COLUMNS:
                features.update(
                    self._get_sequence_feature_columns_input(raw_features,
                                                             'sequence_embedding_column_with_hash_' + key,
                                                             self.sequence_embedding_feature_column_with_hash[key]))

        # sequence embedding file
        if len(config.FILE_SEQUENCE_EMBEDDING_COLUMNS) > 0:
            for key, _, _ in config.FILE_SEQUENCE_EMBEDDING_COLUMNS:
                features.update(
                    self._get_sequence_feature_columns_input(raw_features,
                                                             'sequence_embedding_column_with_vocabulary_file_' + key,
                                                             self.sequence_embedding_feature_column_with_vocabulary_file[
                                                                 key]))

        # multi-hot hash
        if len(config.HASH_MULTI_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.HASH_MULTI_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_multi_hot_feature_columns_input(raw_features,
                                                              'multi_hot_column_with_hash_' + key,
                                                              # multi_categorical_column_with_hash_
                                                              self.multi_categorical_feature_column_with_hash[key]))

        # multi-hot file
        if len(config.FILE_MULTI_CATEGORICAL_COLUMNS) > 0:
            for key, _ in config.FILE_MULTI_CATEGORICAL_COLUMNS:
                features.update(
                    self._get_multi_hot_feature_columns_input(raw_features,
                                                              'multi_hot_column_with_vocabulary_file_' + key,
                                                              # multi_categorical_column_with_vocabulary_file_
                                                              self.multi_categorical_feature_column_with_vocabulary_file[
                                                                  key]))

        # ============================ add more feature start ============================
        #  specific feature columns process can add here
        if len(config.LOCAL_VECS) > 0:
            for key, vec_dim, meta_data_path, vec_data_path in config.LOCAL_VECS:
                self._load_vec_feature(features, raw_features, key, vec_dim, meta_data_path, vec_data_path)
        # ============================ add more feature end ============================

        return features

    # ------------------------ chain all act ------------------------

    def _build(self):

        self._get_numerical_feature_column()

        self._get_bucketized_feature_column()

        # one-hot
        self._get_categorical_feature_column_with_hash()
        self._get_categorical_feature_column_with_vocabulary_file()

        # embedding
        self._get_embedding_feature_column_with_hash()
        self._get_embedding_feature_column_with_vocabulary_file()

        # sequence one-hot
        self._get_sequence_categorical_feature_column_with_hash()
        self._get_sequence_categorical_feature_column_with_vocabulary_file()

        # sequence embedding
        self._get_sequence_embedding_feature_column_with_hash()
        self._get_sequence_embedding_feature_column_with_vocabulary_file()

        # multi-hot
        self._get_multi_categorical_feature_column_with_hash()
        self._get_multi_categorical_feature_column_with_vocabulary_file()

    def gather_feature_columns(self):
        feature_columns = []

        feature_columns += self.numerical_feature_column

        feature_columns += self.bucketized_feature_column

        # one-hot
        feature_columns += self.categorical_feature_column_with_hash.values()
        feature_columns += self.categorical_feature_column_with_vocabulary_file.values()

        # embedding
        feature_columns += self.embedding_feature_column_with_hash.values()
        feature_columns += self.embedding_feature_column_with_vocabulary_file.values()

        # sequence one-hot
        feature_columns += self.sequence_categorical_feature_column_with_hash.values()
        feature_columns += self.sequence_categorical_feature_column_with_vocabulary_file.values()

        # sequence embedding
        feature_columns += self.sequence_embedding_feature_column_with_hash.values()
        feature_columns += self.sequence_embedding_feature_column_with_vocabulary_file.values()

        # multi-hot
        feature_columns += self.multi_categorical_feature_column_with_hash.values()
        feature_columns += self.multi_categorical_feature_column_with_vocabulary_file.values()

        return feature_columns

    @staticmethod
    def _load_local_vec_file(meta_data_path, vec_data_path):
        meta_data_path = meta_data_path
        vec_data_path = vec_data_path
        keys = []
        values = []
        with open(meta_data_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                id = line.strip()
                keys.append(id)
        with open(vec_data_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                vec = line.strip()
                values.append(vec)
        default_val = ''
        table = HashTable(
            KeyValueTensorInitializer(keys, values), default_val)
        # table.init.run(session=self.sess)
        return table

    @staticmethod
    def _string_vec_to_num_vec(string_tensor, vec_dim):
        num_vec = tf.reshape(tf.string_to_number(
            tf.sparse_tensor_to_dense(tf.string_split(string_tensor, ","), default_value=''), tf.float32),
            [-1, vec_dim])
        return num_vec

    def _load_vec_feature(self, features, raw_features, key, vec_dim, meta_data_path, vec_data_path):
        vec_table = self._load_local_vec_file(meta_data_path, vec_data_path)
        features.update({'local_vec_' + key: self._string_vec_to_num_vec(vec_table.lookup(raw_features[key]), vec_dim)})

    def _build_hash_table(self, map_file, map_cols, sep):
        # first col of map_file must be id
        keys = []
        values = dict(zip(map_cols, [[] for _ in range(len(map_cols))]))

        with open(map_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line_split = line.strip().split(sep)
                keys.append(line_split[0].encode('utf8'))
                for i, col in enumerate(map_cols):
                    values[col].append(line_split[i + 1].encode('utf8'))

        for col in map_cols:
            default_val = '-'
            table = HashTable(
                KeyValueTensorInitializer(keys, values[col]), default_val)
            self.hash_table[col] = table

    def _extend_more_feature(self, raw_features, key, map_cols):
        for col in map_cols:
            raw_features[key + '_' + col] = self.hash_table[col].lookup(raw_features[key])
        return raw_features


if __name__ == '__main__':

    with tf.Session() as sess:

        FeaturesProcess = Features()

        raw_features, labels = input_fn(data_file=config.DATA_PATH + '/train',
                                        num_epochs=2,
                                        mode='eval',
                                        batch_size=3)

        feature = FeaturesProcess.get_processed_feature(raw_features)

        f = FeaturesProcess.gather_feature_columns()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # print(f)
        # print('\n')

        a, b, c = sess.run([raw_features, feature, labels])

    for k, v in b.items():
        print(k)
        print(v.shape)
        print(v)
        print('\n')

    print(a, c)
