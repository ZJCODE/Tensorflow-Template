# -*- coding: utf-8 -*-
"""
Created on 2018.8.8

@author: zhangjun
"""


import tensorflow as tf

def embedding_layer(id_feature_name,cate_size,embedding_size,feature_name):
    with tf.variable_scope(feature_name):    
        embedding_matrix = tf.get_variable(name = "embedding_matrix",
                                   shape = [cate_size,embedding_size],
                                   initializer = tf.truncated_normal_initializer(),
                                   dtype =tf.float32)   

        embedding_vec = tf.nn.embedding_lookup(params = embedding_matrix,
                                               ids = id_feature_name,
                                               name = "embedding_vec")
    return embedding_matrix,embedding_vec


def fully_connection_layer(input_data,layer_size,regular_rate = 0 ,dropout = 0,training = True):
    if regular_rate:
        l2_regular = tf.contrib.layers.l2_regularizer(regular_rate)
    else:
        l2_regular = None

    output_data = tf.layers.dense(inputs = input_data,
                                  units = layer_size,
                                  activation=tf.nn.relu,
                                  kernel_regularizer = l2_regular)

    output_data = tf.layers.batch_normalization(inputs = output_data,
                                                training = training)
    
    if dropout:            
        output_data = tf.layers.dropout(inputs = output_data, 
                                        rate = dropout, 
                                        training = training, 
                                        name='dropout')
    return output_data


def multilayer_perceptron_layer(input_data,layer_size_list,regular_rate = 0 ,dropout = 0 ,training = True):
    for layer_size in layer_size_list:
        input_data = fully_connection_layer(input_data,layer_size,regular_rate,dropout,training)
    output_data = input_data
    return output_data


def get_optimizer(opt_algo, learning_rate, loss):
    if opt_algo == 'adagrad':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    elif opt_algo == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    
    else:
        print("input error use adam for default")
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)


def get_loss(labels,logits,type = 'sigmoid_cross_entropy'):

    if type == 'sigmoid_cross_entropy':
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels = labels, # target integer labels in {0, 1}
                                                              logits = logits))
    elif type == 'softmax_cross_entropy':
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = labels, # One-hot-encoded labels.
                                                              logits = logits))
    elif type == 'sparse_softmax_cross_entropy':
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels = labels, # labels must be an index in [0, num_classes)
                                                                     logits = logits))

    elif type == 'weighted_cross_entropy':
        loss = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(targets = labels,
                                                 logits = logits,
                                                  pos_weight = 1.0,
                                                  name='weighted_cross_entropy')))
    else:
      loss = tf.constant(0.0)
    return loss


# def lstm_layer(inputs,batch_size,num_steps,hidden_size,num_layers,dropout_rate,is_training):

#     # inputs [batch_size, num_steps, input_size]
#     cell = tf.contrib.rnn.MultiRNNCell(
#         [_make_cell(hidden_size,dropout_rate,is_training) for _ in range(num_layers)], state_is_tuple=True)

#     # initial_state (LSTMStateTuple,LSTMStateTuple,...) len is num_layers
#     initial_state = cell.zero_state(batch_size, tf.float32)

#     # inputs_unstack [[batch_size,input_size],[batch_size,input_size],...] len is num_steps 
#     inputs_unstack = tf.unstack(inputs, num=num_steps, axis=1)

#     outputs, state = tf.nn.static_rnn(cell, inputs_unstack,
#                                       initial_state=initial_state)

#     # outputs [[batch_size,hidden_size],[batch_size,hidden_size],...] len is num_steps 
#     # state (LSTMStateTuple,LSTMStateTuple,...) len is num_layers
#     return outputs, state


# def _get_lstm_cell(hidden_size,is_training):
#     lstm_cell= tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size,
#                                       forget_bias=0.0, 
#                                       state_is_tuple=True,
#                                       reuse=not is_training)
#     return lstm_cell

# def _make_cell(hidden_size,dropout_rate,is_training):
#     cell = _get_lstm_cell(hidden_size,is_training)
#     if is_training and dropout_rate >0 :
#         cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1 - dropout_rate)
#     return cell


def dynamic_lstm_layer(inputs,sequence_length,hidden_size_list):

    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden_size_list]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    # 'outputs' is a tensor of shape [batch_size, max_time, hidden_size_list[-1]]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                       inputs=inputs,
                                       sequence_length=sequence_length,
                                       dtype=tf.float32)

    return outputs,state


def cnn_layer():
    pass




