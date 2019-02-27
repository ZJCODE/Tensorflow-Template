# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
import numpy as np


def fully_connection(inputs, units, regular_rate=0.0, dropout=0.0, training=True):
    """
    do not forget control_dependencies

     x_norm = tf.layers.batch_normalization(x, training=training)

      # ...

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    """

    output_data = tf.layers.dense(inputs=inputs,
                                  units=units,
                                  activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_rate))

    output_data = tf.layers.batch_normalization(inputs=output_data,
                                                training=training)

    output_data = tf.layers.dropout(inputs=output_data,
                                    rate=dropout,
                                    training=training,
                                    name='dropout')
    return output_data


def mlp(inputs, hidden_units_list, regular_rate=0.0, dropout=0.0, training=True):
    for units in hidden_units_list:
        inputs = fully_connection(inputs, units, regular_rate, dropout, training)
    output_data = inputs
    return output_data


def dynamic_rnn(sequence_vector, sequence_real_len, units_num, cell_type='gru'):
    """
    sequence_vector: [batch_size,sequence_length,vector_size]
    sequence_real_len: [batch_size,]
    num_units: int number
    cell_type: GRU or LSTM
    output_type: last or all
    """

    if len(sequence_real_len.get_shape().as_list()) == 2:
        sequence_real_len = tf.squeeze(sequence_real_len, axis=1)

    if cell_type.lower() == 'gru':
        cell = tf.nn.rnn_cell.GRUCell(units_num)
    elif cell_type.lower() == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(units_num)
    else:
        print("use default GRU cell")
        cell = tf.nn.rnn_cell.GRUCell(units_num)

    input_data = tf.transpose(sequence_vector, [1, 0, 2])

    outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                        inputs=input_data,
                                        sequence_length=sequence_real_len,
                                        dtype=tf.float32,
                                        time_major=True)

    outputs = tf.transpose(outputs, [1, 0, 2])

    #     batch_size = tf.shape(outputs)[0]
    #     index = tf.range(0, batch_size) * sequence_length + (real_length - 1)
    #     outputs_last = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    #     outputs_last is same as states

    return outputs, states


def sequence_mask(sequence_vector, sequence_real_len):
    """
    :param sequence_vector:
        shape : [batch_size,sequence_length,vector_size]
        example : [[[2,3,3],[4,3,1],[4,2,1],[4,2,1]],[[2,3,5],[5,4,3],[4,2,2],[4,2,2]]]
    :param sequence_real_len:
        shape :[batch_size,]
        example : [3,1]
    :return: sequence_data_masked
        shape : [batch_size,sequence_len,vec_size]
        example : [[[2,3,3],[4,3,1],[4,2,1],[0,0,0]],[[2,3,5],[0,0,0],[0,0,0],[0,0,0]]]

    """
    sequence_vector = tf.cast(sequence_vector, tf.float32)
    sequence_len = sequence_vector.get_shape().as_list()[-2]
    mask = tf.sequence_mask(tf.squeeze(sequence_real_len), sequence_len)
    mask2value = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
    sequence_data_masked = tf.multiply(sequence_vector, mask2value)
    return sequence_data_masked


def matmul_3d_with_2d(tensor_3d, tensor_2d):
    """
    :param tensor_3d: usually is embedding vec for sequence
        shape: [batch_size,sequence_len,vec_size]
    :param tensor_2d: usually is weight
        shape: [vec_size,vec_size_new]
    :return: tensor_3d_new
        shape: [batch_size,sequence_len,vec_size_new]
    """
    return tf.tensordot(tensor_3d, tensor_2d, axes=[[2], [0]])


def soft_filter_noise(sequence_vector):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :return: [batch_size,sequence_length,vector_size]

    # Example :
    # inputs = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """
    normalized_sequence_vector = tf.nn.l2_normalize(sequence_vector, axis=2)
    sim_cos = tf.matmul(normalized_sequence_vector, tf.transpose(normalized_sequence_vector, perm=[0, 2, 1]))
    important_weight = tf.reduce_mean(sim_cos, axis=2, keepdims=True)
    weighted_sequence_vector = tf.multiply(sequence_vector, important_weight)
    return weighted_sequence_vector


def soft_enhance_noise(sequence_vector):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :return: [batch_size,sequence_length,vector_size]

    # Example :
    # inputs = tf.constant([[[1.0,2],[-4,3],[2,3]],[[1,2],[4,3],[-2,3]]])
    """
    normalized_sequence_vector = tf.nn.l2_normalize(sequence_vector, axis=2)
    sim_cos = tf.matmul(normalized_sequence_vector, tf.transpose(normalized_sequence_vector, perm=[0, 2, 1]))
    important_weight = 1.0 / tf.reduce_mean(sim_cos, axis=2, keepdims=True)
    weighted_sequence_vector = tf.multiply(sequence_vector, important_weight)
    return weighted_sequence_vector


def add_attention_weight(vector, sequence_vector, attention_type='dot', with_softmax=False, l2_normalize=False):
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

        if l2_normalize:
            # [batch_size,vector_size,1]
            normalized_vector_expand = tf.nn.l2_normalize(tf.expand_dims(vector, axis=2), axis=1)
            # [batch_size,sequence_length,vector_size]
            normalized_sequence_vector = tf.nn.l2_normalize(sequence_vector, axis=2)
        else:
            normalized_vector_expand = tf.expand_dims(vector, axis=2)
            normalized_sequence_vector = sequence_vector
        # [batch_size,sequence_length,1]
        sim_cos_weight = tf.matmul(normalized_sequence_vector, normalized_vector_expand)
        if with_softmax:
            sim_cos_weight_squeeze = tf.squeeze(sim_cos_weight, axis=2)
            sim_cos_weight_squeeze = tf.nn.softmax(sim_cos_weight_squeeze, axis=1)
            sim_cos_weight = tf.expand_dims(sim_cos_weight_squeeze, axis=2)
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
            mlp_attention_weight_squeeze = tf.squeeze(mlp_attention_weight, axis=2)
            mlp_attention_weight_squeeze = tf.nn.softmax(mlp_attention_weight_squeeze, axis=1)
            mlp_attention_weight = tf.expand_dims(mlp_attention_weight_squeeze, axis=2)

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


def sequence_conv_pooling(sequence_vector, kernel_size_first_dim=3, output_dim=0, pooling_type='max'):
    """
    :param sequence_vector: [batch_size,sequence_length,vector_size]
    :param kernel_size_first_dim: how many vectors considered at one time when doing convolution
    :return: sequence_vector_conv_avg [batch_size,output_dim] # default set output dim to log(vector_size)

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
        sequence_vector_conv_pooling = tf.squeeze(tf.reduce_max(sequence_vector_conv, axis=1), axis=1)
    elif pooling_type == 'mean':
        sequence_vector_conv_pooling = tf.squeeze(tf.reduce_mean(sequence_vector_conv, axis=1), axis=1)
    else:
        sequence_vector_conv_pooling = tf.squeeze(tf.reduce_max(sequence_vector_conv, axis=1), axis=1)
    return sequence_vector_conv_pooling


def padding_clip_sequence_vector(sequence_vector, clip_length):
    """
    :param sequence_vector: [None,None,vec_dim] [batch_size,batch_max_seq_length,vec_dim]
    :param clip_length:
    :return:
    """
    vec_dim = sequence_vector.get_shape().as_list()[-1]
    sequence_vector = tf.pad(sequence_vector, [[0, 0], [0, clip_length], [0, 0]], mode='CONSTANT')[:, :clip_length, :]
    sequence_vector = tf.reshape(sequence_vector, [-1, clip_length, vec_dim])
    return sequence_vector


def get_identity_matrix(inputs):
    """
    inputs: [vec_dim,vec_dim]
    """
    return tf.to_float(tf.matrix_diag(tf.diag_part(tf.ones_like(inputs))))


def get_identity_matrix_batch(inputs):
    """
    inputs: [batch_size,vec_dim,vec_dim]
    """
    return tf.to_float(tf.matrix_set_diag(tf.zeros_like(inputs), tf.matrix_diag_part(tf.ones_like(inputs))))


if __name__ == '__main__':
    sess = tf.Session()
    a = tf.constant([[[2, 3.0, 3], [4, 3, 1], [4, 2, 1], [4, 2, 1]], [[2, 3, 5], [5, 4, 3], [4, 2, 2], [4, 2, 2]]])
    b = tf.constant([3, 1])
    c = tf.constant([[2, 3.0, 3], [2, 3.0, 3]])

    # print(sess.run(sequence_mask(a, b)))
    # print(sess.run(soft_enhance_noise(a)))
    # print(sess.run(soft_filter_noise(a)))

    o, s = dynamic_rnn(a, b, 3, cell_type='gru')
    # oo = sequence_conv_pooling(a, kernel_size_first_dim=3, output_dim=10, pooling_type='max')
    # ooo, _ = add_attention_weight(c, a, attention_type='dot', with_softmax=True, l2_normalize=False)
    sess.run(tf.global_variables_initializer())

    x, w = sess.run([o, s])
    # y = sess.run(oo)
    # z = sess.run(ooo)
    print(a)
    print(x)
    print('-' * 10)
    print(w)
    # print(y)
    # print('-' * 10)
    # print(z)
