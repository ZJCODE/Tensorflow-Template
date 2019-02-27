import tensorflow as tf
import numpy as np
from tensorflow.contrib.lookup import HashTable
from tensorflow.contrib.lookup import KeyValueTensorInitializer
from tensorflow.contrib.feature_column import sequence_categorical_column_with_hash_bucket
from tensorflow.contrib.feature_column import sequence_input_layer
from tensorflow.contrib.feature_column import sequence_categorical_column_with_vocabulary_file


def embedding_dim(dim):
    """empirical embedding dim"""
    return int(np.power(2, np.ceil(np.log(dim ** 0.25))))


def share_embedding(feature_dict, feature_name_list, vocabulary_path, embedding_size, weight_name, oov=None):
    vocabularys = list(np.loadtxt(vocabulary_path, dtype='str', encoding='utf8'))
    vocabulary_size = len(vocabularys)

    if oov == None:
        indexs = list(range(1, len(vocabularys) + 1))  # leave 0 for default
        vocabulary_index_map = HashTable(KeyValueTensorInitializer(vocabularys, indexs), 0)
        vocabulary_size = vocabulary_size + 1
    elif oov == 0:
        indexs = list(range(0, len(vocabularys)))
        vocabulary_index_map = HashTable(KeyValueTensorInitializer(vocabularys, indexs), 0)
        vocabulary_size = vocabulary_size
    else:
        AssertionError("oov euqal to None or 0")

    if embedding_size == -1:  # dim_auto
        embedding_size = embedding_dim(vocabulary_size)

    embedding_weight = tf.get_variable(name=weight_name,
                                       initializer=tf.glorot_normal_initializer(),
                                       shape=[vocabulary_size, embedding_size])

    embeddings = []
    for feature_name in feature_name_list:
        embeddings.append(
            tf.nn.embedding_lookup(embedding_weight, vocabulary_index_map.lookup(feature_dict[feature_name])))

    return embeddings


def get_numeric_column(feature_dict, feature_name):
    if isinstance(feature_name, list):
        numerical_feature_column = []
        for name in feature_name:
            feature_col = tf.feature_column.numeric_column(name)
            numerical_feature_column.append(feature_col)
    else:
        numerical_feature_column = tf.feature_column.numeric_column(feature_name)
    numeric_column = tf.feature_column.input_layer(feature_dict, numerical_feature_column)
    return numeric_column


def get_bucketized_column(feature_dict, feature_name, boundaries):
    if isinstance(feature_name, list) and isinstance(boundaries, list):
        bucketized_feature_column = []
        for name, boundary in zip(feature_name, boundaries):
            feature_col = tf.feature_column.numeric_column(name)
            feature_col = tf.feature_column.bucketized_column(feature_col, boundary)
            bucketized_feature_column.append(feature_col)
    else:
        feature_col = tf.feature_column.numeric_column(feature_name)
        bucketized_feature_column = tf.feature_column.bucketized_column(feature_col, boundaries)
    bucketized_column = tf.feature_column.input_layer(feature_dict, bucketized_feature_column)
    return bucketized_column


def get_one_hot_column_with_hash(feature_dict, feature_name, hash_bucket_size):
    feature_col = tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.string)
    one_hot_feature_column_with_hash = tf.feature_column.indicator_column(feature_col)
    one_hot_column_with_hash = tf.feature_column.input_layer(feature_dict, one_hot_feature_column_with_hash)
    return one_hot_column_with_hash


def get_one_hot_column_with_vocabulary_file(feature_dict, feature_name, vocabulary_path):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    feature_col = tf.feature_column.categorical_column_with_vocabulary_file(feature_name, vocabulary_path,
                                                                            vocabulary_size)
    one_hot_feature_column_with_vocabulary_file = tf.feature_column.indicator_column(feature_col)
    one_hot_column_with_vocabulary_file = tf.feature_column.input_layer(feature_dict,
                                                                        one_hot_feature_column_with_vocabulary_file)
    return one_hot_column_with_vocabulary_file


def get_embedding_column_with_hash(feature_dict, feature_name, hash_bucket_size, embedding_size):
    if embedding_size == -1:  # dim_auto
        embedding_size = embedding_dim(hash_bucket_size)
    feature_col = tf.feature_column.categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.string)
    embedding_feature_column_with_hash = tf.feature_column.embedding_column(feature_col, embedding_size)
    embedding_column_with_hash = tf.feature_column.input_layer(feature_dict, embedding_feature_column_with_hash)
    return embedding_column_with_hash


def get_embedding_column_with_vocabulary_file(feature_dict, feature_name, vocabulary_path, embedding_size):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    if embedding_size == -1:  # dim_auto
        embedding_size = embedding_dim(vocabulary_size)
    feature_col = tf.feature_column.categorical_column_with_vocabulary_file(feature_name, vocabulary_path,
                                                                            vocabulary_size)
    embedding_feature_column_with_vocabulary_file = tf.feature_column.embedding_column(feature_col, embedding_size)
    embedding_column_with_vocabulary_file = tf.feature_column.input_layer(feature_dict,
                                                                          embedding_feature_column_with_vocabulary_file)
    return embedding_column_with_vocabulary_file


def get_sequence_one_hot_column_with_hash(feature_dict, feature_name, hash_bucket_size):
    feature_col = sequence_categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.string)
    sequence_one_hot_feature_column_with_hash = tf.feature_column.indicator_column(feature_col)
    sequence_one_hot_column_with_hash = sequence_input_layer(feature_dict, sequence_one_hot_feature_column_with_hash)[0]
    return sequence_one_hot_column_with_hash


def get_sequence_one_hot_column_with_vocabulary_file(feature_dict, feature_name, vocabulary_path):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    feature_col = sequence_categorical_column_with_vocabulary_file(feature_name, vocabulary_path, vocabulary_size)
    sequence_one_hot_feature_column_with_vocabulary_file = tf.feature_column.indicator_column(feature_col)
    sequence_one_hot_column_with_vocabulary_file = sequence_input_layer(
        feature_dict,
        sequence_one_hot_feature_column_with_vocabulary_file)[0]
    return sequence_one_hot_column_with_vocabulary_file


def get_sequence_embedding_column_with_hash(feature_dict, feature_name, hash_bucket_size, embedding_size):
    if embedding_size == -1:  # dim_auto
        embedding_size = embedding_dim(hash_bucket_size)
    feature_col = sequence_categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.string)
    sequence_embedding_feature_column_with_hash = tf.feature_column.embedding_column(feature_col, embedding_size)
    sequence_embedding_column_with_hash = sequence_input_layer(
        feature_dict,
        sequence_embedding_feature_column_with_hash)[0]
    return sequence_embedding_column_with_hash


def get_sequence_embedding_column_with_vocabulary_file(feature_dict, feature_name, vocabulary_path, embedding_size):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    if embedding_size == -1:  # dim_auto
        embedding_size = embedding_dim(vocabulary_size)
    feature_col = sequence_categorical_column_with_vocabulary_file(feature_name, vocabulary_path, vocabulary_size)
    sequence_embedding_feature_column_with_vocabulary_file = tf.feature_column.embedding_column(feature_col,
                                                                                                embedding_size)
    sequence_embedding_column_with_vocabulary_file = sequence_input_layer(
        feature_dict,
        sequence_embedding_feature_column_with_vocabulary_file)[0]
    return sequence_embedding_column_with_vocabulary_file


def get_multi_categorical_column_with_hash(feature_dict, feature_name, hash_bucket_size):
    feature_col = sequence_categorical_column_with_hash_bucket(feature_name, hash_bucket_size, dtype=tf.string)
    multi_categorical_feature_column_with_hash = tf.feature_column.indicator_column(feature_col)
    multi_categorical_column_with_hash = tf.reduce_sum(
        sequence_input_layer(feature_dict, multi_categorical_feature_column_with_hash)[0], axis=1)
    return multi_categorical_column_with_hash


def get_multi_categorical_column_with_vocabulary_file(feature_dict, feature_name, vocabulary_path):
    vocabulary_size = len(open(vocabulary_path, 'r', encoding='utf8').readlines())
    feature_col = sequence_categorical_column_with_vocabulary_file(feature_name, vocabulary_path, vocabulary_size)
    multi_categorical_feature_column_with_vocabulary_file = tf.feature_column.indicator_column(feature_col)
    multi_categorical_column_with_vocabulary_file = tf.reduce_sum(
        sequence_input_layer(feature_dict, multi_categorical_feature_column_with_vocabulary_file)[0], axis=1)
    return multi_categorical_column_with_vocabulary_file


def get_local_vec(feature_dict, feature_name, meta_data_path, vec_data_path):
    keys = list(np.loadtxt(meta_data_path, dtype='str', encoding='utf8'))
    indexs = list(range(1, len(keys) + 1))  # leave 0 for default

    vec_index_map = HashTable(
        KeyValueTensorInitializer(keys, indexs), 0)

    value_array = np.loadtxt(vec_data_path, dtype=np.float32, delimiter=',', encoding='utf8')
    default_val = value_array.mean(0)
    vec_var_for_initial = np.vstack([default_val, value_array])  # leave 0 [first line] for default

    vec_var = tf.get_variable(name=feature_name + '_local_vec',
                              initializer=vec_var_for_initial,
                              trainable=False,
                              dtype=tf.float32)

    feature_local_vec = tf.nn.embedding_lookup(vec_var, vec_index_map.lookup(feature_dict[feature_name]))

    return feature_local_vec
