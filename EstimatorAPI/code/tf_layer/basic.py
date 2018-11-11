# -*- coding: utf-8 -*-
"""
Created on 2018.10.25

@author: zhangjun
"""

import tensorflow as tf


def fully_connection_layer(input_data, layer_size, regular_rate=0.0, dropout=0.0, training=True):
    output_data = tf.layers.dense(inputs=input_data,
                                  units=layer_size,
                                  activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(regular_rate))

    output_data = tf.layers.batch_normalization(inputs=output_data,
                                                training=training)

    output_data = tf.layers.dropout(inputs=output_data,
                                    rate=dropout,
                                    training=training,
                                    name='dropout')
    return output_data


def multilayer_perceptron_layer(input_data, layer_size_list, regular_rate=0, dropout=0, training=True):
    for layer_size in layer_size_list:
        input_data = fully_connection_layer(input_data, layer_size, regular_rate, dropout, training)
    output_data = input_data
    return output_data
