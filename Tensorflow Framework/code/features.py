# -*- coding: utf-8 -*-
"""
Created on 2018.12.18

@author: zhangjun
"""

import tensorflow as tf
import numpy as np

from utils.conf import Config
from utils.feature import *

config = Config(base_dir='../conf')


def get_processed_feature(raw_features):
    # raw_features the same sa result returned by input_fn [see the type in input_fn ]
    features = {}
    features.update(raw_features)

    if len(config.NUMERICAL_COLUMNS) > 0:
        features.update({'numerical_columns': get_numeric_column(raw_features, config.NUMERICAL_COLUMNS)})
        for feature_name in config.NUMERICAL_COLUMNS:
            features.update({'numerical_columns_' + feature_name: get_numeric_column(raw_features, feature_name)})

    if len(config.BUCKET_COLUMNS) > 0:
        feature_names = [x[0] for x in config.BUCKET_COLUMNS]
        boundaries = [x[1] for x in config.BUCKET_COLUMNS]
        features.update({'bucketized_column': get_bucketized_column(raw_features, feature_names, boundaries)})
        for feature_name, boundary in config.BUCKET_COLUMNS:
            features.update(
                {'bucketized_column_' + feature_name: get_bucketized_column(raw_features, feature_name, boundary)})

    # one-hot hash
    if len(config.HASH_CATEGORICAL_COLUMNS) > 0:
        for feature_name, hash_bucket_size in config.HASH_CATEGORICAL_COLUMNS:
            features.update({'one_hot_column_with_hash_' + feature_name:
                                 get_one_hot_column_with_hash(raw_features, feature_name, hash_bucket_size)})

    # one-hot file
    if len(config.FILE_CATEGORICAL_COLUMNS) > 0:
        for feature_name, vocabulary_path in config.FILE_CATEGORICAL_COLUMNS:
            features.update(
                {'one_hot_column_with_vocabulary_file_' + feature_name:
                     get_one_hot_column_with_vocabulary_file(raw_features, feature_name, vocabulary_path)})

    # embedding hash
    if len(config.HASH_EMBEDDING_COLUMNS) > 0:
        for feature_name, hash_bucket_size, embedding_size in config.HASH_EMBEDDING_COLUMNS:
            features.update(
                {'embedding_column_with_hash_' + feature_name:
                     get_embedding_column_with_hash(raw_features, feature_name, hash_bucket_size, embedding_size)})

    # embedding file
    if len(config.FILE_EMBEDDING_COLUMNS) > 0:
        for feature_name, vocabulary_path, embedding_size in config.FILE_EMBEDDING_COLUMNS:
            features.update(
                {'embedding_column_with_vocabulary_file_' + feature_name:
                     get_embedding_column_with_vocabulary_file(raw_features, feature_name, vocabulary_path,
                                                               embedding_size)})

    # sequence ont-hot hash
    if len(config.HASH_SEQUENCE_CATEGORICAL_COLUMNS) > 0:
        for feature_name, hash_bucket_size in config.HASH_SEQUENCE_CATEGORICAL_COLUMNS:
            features.update(
                {'sequence_one_hot_column_with_hash_' + feature_name:
                     get_sequence_one_hot_column_with_hash(raw_features, feature_name, hash_bucket_size)})

    # sequence ont-hot file
    if len(config.FILE_SEQUENCE_CATEGORICAL_COLUMNS) > 0:
        for feature_name, vocabulary_path in config.FILE_SEQUENCE_CATEGORICAL_COLUMNS:
            features.update(
                {'sequence_one_hot_column_with_vocabulary_file_' + feature_name:
                     get_sequence_one_hot_column_with_vocabulary_file(raw_features, feature_name, vocabulary_path)})

    # sequence embedding hash
    if len(config.HASH_SEQUENCE_EMBEDDING_COLUMNS) > 0:
        for feature_name, hash_bucket_size, embedding_size in config.HASH_SEQUENCE_EMBEDDING_COLUMNS:
            features.update(
                {'sequence_embedding_column_with_hash_' + feature_name:
                     get_sequence_embedding_column_with_hash(raw_features, feature_name, hash_bucket_size,
                                                             embedding_size)})

    # sequence embedding file
    if len(config.FILE_SEQUENCE_EMBEDDING_COLUMNS) > 0:
        for feature_name, vocabulary_path, embedding_size in config.FILE_SEQUENCE_EMBEDDING_COLUMNS:
            features.update(
                {'sequence_embedding_column_with_vocabulary_file_' + feature_name:
                     get_sequence_embedding_column_with_vocabulary_file(raw_features, feature_name, vocabulary_path,
                                                                        embedding_size)})

    # multi-hot hash
    if len(config.HASH_MULTI_CATEGORICAL_COLUMNS) > 0:
        for feature_name, hash_bucket_size in config.HASH_MULTI_CATEGORICAL_COLUMNS:
            features.update(
                {'multi_hot_column_with_hash_' + feature_name:
                     get_multi_categorical_column_with_hash(raw_features, feature_name, hash_bucket_size)})

    # multi-hot file
    if len(config.FILE_MULTI_CATEGORICAL_COLUMNS) > 0:
        for feature_name, vocabulary_path in config.FILE_MULTI_CATEGORICAL_COLUMNS:
            features.update(
                {'multi_hot_column_with_vocabulary_file_' + feature_name:
                     get_multi_categorical_column_with_vocabulary_file(raw_features, feature_name,
                                                                       vocabulary_path)})

    if len(config.LOCAL_VECS) > 0:
        for feature_name, meta_data_path, vec_data_path in config.LOCAL_VECS:
            features.update(
                {'local_vec_' + feature_name:
                     get_local_vec(raw_features, feature_name, meta_data_path, vec_data_path)})
    return features


if __name__ == '__main__':
    pass
