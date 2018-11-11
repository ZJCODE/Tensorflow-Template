# -*- coding: utf-8 -*-
"""
Created on 2018.10.14

@author: zhangjun
"""
import tensorflow as tf
import numpy as np


def soft_filter_noise_vector_old(sequence_vector):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :return: [batch_size,sequence_length,vector_size]

    # Example :
    # sequence_vector = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """
    sim_dot = tf.matmul(sequence_vector, tf.transpose(sequence_vector, perm=[0, 2, 1]))
    sequence_vector_norm = tf.norm(sequence_vector, axis=2, keepdims=True)
    divide_norm = tf.matmul(sequence_vector_norm, tf.transpose(sequence_vector_norm, perm=[0, 2, 1]))
    sim_cos = tf.divide(sim_dot, tf.add(divide_norm, tf.constant(1e-8)))
    important_weight = tf.reduce_mean(sim_cos, axis=2, keepdims=True)
    weighted_sequence_vector = tf.multiply(sequence_vector, important_weight)
    return weighted_sequence_vector


def soft_filter_noise_vector(sequence_vector):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :return: [batch_size,sequence_length,vector_size]

    # Example :
    # sequence_vector = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """
    normalized_sequence_vector = tf.nn.l2_normalize(sequence_vector, axis=2)
    sim_cos = tf.matmul(normalized_sequence_vector, tf.transpose(normalized_sequence_vector, perm=[0, 2, 1]))
    important_weight = tf.reduce_mean(sim_cos, axis=2, keepdims=True)
    weighted_sequence_vector = tf.multiply(sequence_vector, important_weight)
    return weighted_sequence_vector


def add_attention_weight(vector, sequence_vector, attention_type='dot', with_softmax=False):
    """
    :param vector: [batch_size,vector_size]
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :param attention_type: 'dot' or 'mlp'

    :return: attention_weighted_sequence_vector [batch_size,sequence_length,vector_size]
    :return: weight [batch_size,sequence_length,1]

    # Example :
    # vector = tf.constant([[1.0,2],[4,3]])
    # sequence_vector = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """

    if attention_type == 'dot':
        # Denote vector as Q and sequence_vector as V
        # weight_i = cos(Q,V_i)
        # Output [weight_i*V_i,...]

        # [batch_size,vector_size,1]
        normalized_vector_expand = tf.nn.l2_normalize(tf.expand_dims(vector, axis=2), axis=1)
        # [batch_size,sequence_length,vector_size]
        normalized_sequence_vector = tf.nn.l2_normalize(sequence_vector, axis=2)
        # [batch_size,sequence_length,1]
        sim_cos_weight = tf.matmul(normalized_sequence_vector, normalized_vector_expand)
        if with_softmax:
            sim_dot_weight = tf.nn.softmax(sim_cos_weight, axis=1)
        attention_weighted_sequence_vector = tf.multiply(sequence_vector, sim_cos_weight)
        weight = sim_cos_weight
    elif attention_type == 'mlp':
        # Denote vector as Q and sequence_vector as V
        # weight_i = W_2^T·W_1^T·[Q,V_i]
        # Output [weight_i*V_i,...]

        # [batch_size,1,vector_size]
        vector_expand = tf.expand_dims(vector, axis=1)
        sequence_length = tf.shape(sequence_vector)[1]
        # [batch_size,sequence_length,vector_size]
        vector_expand_tile = tf.tile(vector_expand, multiples=tf.stack([1, sequence_length, 1]))
        # concat_vector_and_sequence_vector [batch_size,sequence_length,vector_size + sequence_vector_i_size]
        concat_vector_and_sequence_vector = tf.concat([vector_expand_tile, sequence_vector], axis=2)
        concat_vector_last_dim = concat_vector_and_sequence_vector.get_shape().as_list()[-1]
        w1 = tf.get_variable(name="attention_mlp_wight_1",
                             shape=[concat_vector_last_dim, int(np.sqrt(concat_vector_last_dim))],
                             initializer=tf.glorot_normal_initializer(),
                             dtype=tf.float32)
        w2 = tf.get_variable(name="attention_mlp_wight_2",
                             shape=[int(np.sqrt(concat_vector_last_dim)), 1],
                             initializer=tf.glorot_normal_initializer(),
                             dtype=tf.float32)
        # mlp_attention_weight [batch_size,sequence_length,int(concat_vector_last_dim/2)]
        mlp_attention_weight = tf.nn.relu(tf.tensordot(concat_vector_and_sequence_vector, w1, axes=[[2], [0]]))
        # mlp_attention_weight [batch_size,sequence_length,1]
        mlp_attention_weight = tf.nn.sigmoid(tf.tensordot(mlp_attention_weight, w2, axes=[[2], [0]]))

        if with_softmax:
            mlp_attention_weight = tf.nn.softmax(mlp_attention_weight, axis=1)
        attention_weighted_sequence_vector = tf.multiply(sequence_vector, mlp_attention_weight)
        weight = mlp_attention_weight
    else:
        # Denote vector as Q and sequence_vector as V
        # weight_i = dot(Q,V_i)/sqrt(vector_size)
        # Output [weight_i*V_i,...]
        vector_expand = tf.expand_dims(vector, axis=2)
        sequence_vector_last_dim = sequence_vector.get_shape().as_list()[-1]
        # sim_dot [batch_size ,sequence_length , 1]
        sim_dot_weight = tf.matmul(sequence_vector, vector_expand) / tf.sqrt(
            tf.constant(sequence_vector_last_dim, dtype=tf.float32))
        if with_softmax:
            sim_dot_weight = tf.nn.softmax(sim_dot_weight, axis=1)
        attention_weighted_sequence_vector = tf.multiply(sequence_vector, sim_dot_weight)
        weight = sim_dot_weight
    return attention_weighted_sequence_vector, weight


def add_multi_attention_weight(vector, sequence_vector, multi_num=5, with_softmax=False):
    """
    :param vector: [batch_size,vector_size]
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :param multi_num: default set 5

    :return: attention_weighted_sequence_vector_multi_concat [batch_size,sequence_length * multi_num,vector_size]
    :return: weight_multi_concat [batch_size,sequence_length * multi_num,1]

    # Example :
    # vector = tf.constant([[1.0,2],[4,3]])
    # sequence_vector = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """
    attention_weighted_sequence_vector_multi = []
    weight_multi = []
    for i in range(multi_num):
        with tf.variable_scope("multi_{}".format(i)):
            attention_weighted_sequence_vector, weight = add_attention_weight(vector=vector,
                                                                              sequence_vector=sequence_vector,
                                                                              attention_type='mlp',
                                                                              with_softmax=with_softmax)
            attention_weighted_sequence_vector_multi.append(attention_weighted_sequence_vector)
            weight_multi.append(weight)
    attention_weighted_sequence_vector_multi_concat = tf.concat(attention_weighted_sequence_vector_multi, axis=1)
    weight_multi_concat = tf.concat(weight_multi, axis=1)
    return attention_weighted_sequence_vector_multi_concat, weight_multi_concat


def sequence_conv_pooling(sequence_vector, kernel_size_first_dim=3, output_dim=0, pooling_type='max'):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :param kernel_size_first_dim: how many vectors considered at one time when doing convolution
    :return: sequence_vector_conv_avg [batch_size,vector_size] # default set output dim to log(vector_size)

    # Example :
    # sequence_vector = tf.constant([[[1.0,2,3,2],[-4,3,7,6],[2,3,2,3]],[[1,2,4,2],[4,3,7,5],[-2,3,2,3]]])
    """
    # notice some sequence only has one object
    # in case of length of sequence less than kernel_size_first_dim, so do padding first
    sequence_vector_pad = tf.pad(sequence_vector, paddings=[[0, 0], [0, kernel_size_first_dim - 1], [0, 0]],
                                 mode='CONSTANT')
    sequence_vector_expand = tf.expand_dims(sequence_vector_pad, axis=3)
    vector_size = sequence_vector.get_shape().as_list()[-1]
    if output_dim <= 0:
        output_dim = int(np.log(vector_size))
    sequence_vector_conv = tf.layers.conv2d(inputs=sequence_vector_expand,
                                            filters=output_dim,
                                            kernel_size=[kernel_size_first_dim, vector_size],
                                            padding='valid',
                                            activation=tf.nn.relu)
    if pooling_type == 'max':
        sequence_vector_conv_avg = tf.squeeze(tf.reduce_max(sequence_vector_conv, axis=1), axis=1)
    elif pooling_type == 'mean':
        sequence_vector_conv_avg = tf.squeeze(tf.reduce_mean(sequence_vector_conv, axis=1), axis=1)
    else:
        sequence_vector_conv_avg = tf.squeeze(tf.reduce_max(sequence_vector_conv, axis=1), axis=1)
    return sequence_vector_conv_avg


def sequence_lstm(sequence_vector):
    pass
