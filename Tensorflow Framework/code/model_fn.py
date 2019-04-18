# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
from utils.model import get_optimizer
from utils.conf import Config
from utils.layer import *
from features import get_processed_feature

config = Config(base_dir='../conf')


def model_fn_core(feature, params):
    # [batch_size,seq_length,vec_dim]
    sequence_vector = feature['sequence_embedding_column_with_vocabulary_file_col6_extend_2']
    numerical = feature['numerical_columns']
    # [batch_size,vec_dim]
    col4 = feature['embedding_column_with_vocabulary_file_col4']

    col7_e1 = feature['embedding_column_with_vocabulary_file_col7_extend_1']
    col7_e2 = feature['embedding_column_with_vocabulary_file_col7_extend_2']

    inputs = tf.concat([tf.reduce_mean(sequence_vector, axis=1), col4, col7_e1, col7_e2, numerical], axis=1)

    # inputs = tf.reduce_mean(sequence_vector, axis=1)

    output = mlp(inputs, hidden_units_list=[16, 1], regular_rate=0.1, dropout=0.1, training=params['training'])

    return output


def get_loss(output, labels):
    labels = tf.reshape(labels, [-1, 1])
    regular_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels))
    loss += regular_loss
    return loss


def get_metric(output, labels):
    """
    :param output:
    :param labels:
    :return: {"metric_name": ''}
    """
    pred = tf.sigmoid(output)
    auc_val = tf.metrics.auc(labels, pred)
    return {"auc": auc_val}


def get_predict(output):
    """
    :param output:
    :return:  {"prob": '')}
    """
    pred = tf.sigmoid(output)
    predictions = {"prob": tf.reshape(pred, [-1])}
    return predictions


def get_export_outputs(output):
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            get_predict(output))}
    return export_outputs


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.TRAIN:
        params.update({"training": True})
    else:
        params.update({"training": False})

    features = get_processed_feature(features)

    output = model_fn_core(features, params)

    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
        loss = get_loss(output, labels)
        eval_metric_ops = get_metric(output, labels)
    else:
        loss = None
        eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # do not forget this | important for batch_normalization
            train_op = get_optimizer(opt_algo=params["opt_algo"],
                                     learning_rate=params["learning_rate"],
                                     loss=loss,
                                     global_step=tf.train.get_global_step())
    else:
        train_op = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = get_predict(output)
        export_outputs = get_export_outputs(output)
    else:
        predictions = None
        export_outputs = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        export_outputs=export_outputs,
        eval_metric_ops=eval_metric_ops)
