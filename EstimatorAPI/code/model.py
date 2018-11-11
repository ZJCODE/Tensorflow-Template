# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

@author: zhangjun
"""

import tensorflow as tf
import time
from datasets import input_fn
from features import Features
from utils.model_util import get_optimizer_instance
from utils.util import elapse_time
import tf_layer.sequence_vector as sv
from tf_layer.basic import multilayer_perceptron_layer


def model_fn(features, labels, mode, params):
    # features, labels return by input_fn [see their type in input_fn ]
    FeaturesProcess = Features()
    # process features
    processed_features = FeaturesProcess.get_processed_feature(features)

    # TODO MODEL STRUCTURE START

    num_input = processed_features['numerical_columns']
    sequence_vector = processed_features['sequence_embedding_column_with_vocabulary_file_col6']  # (3, ?, 5)
    vector = processed_features['embedding_column_with_vocabulary_file_col4']  # (3, 5)

    # weighted_sequence_vector = sv.add_attention_weight(vector, sequence_vector, 'dot')
    weighted_sequence_vector, weight = sv.add_attention_weight(vector, sv.soft_filter_noise_vector(sequence_vector),
                                                               'dot', False)
    # weighted_sequence_vector = sv.add_multi_attention_weight(vector, sv.filter_noise_vector(sequence_vector),
    #                                                           multi_num=5)
    mlp_input = tf.concat([num_input, tf.reduce_mean(weighted_sequence_vector, axis=1)], axis=1)

    sequence_vector_conv_avg = sv.sequence_conv_pooling(sequence_vector=sequence_vector,
                                                        kernel_size_first_dim=3,
                                                        output_dim=32,
                                                        pooling_type='max')

    mlp_input = tf.concat([mlp_input, sequence_vector_conv_avg], axis=1)

    # TODO MODEL STRUCTURE END

    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
    else:
        training = False

    middle = multilayer_perceptron_layer(input_data=mlp_input,
                                         layer_size_list=params['layer_size_list'],
                                         regular_rate=params['regular_rate'],
                                         dropout=params['dropout'],
                                         training=training)

    output = tf.layers.dense(inputs=middle,
                             units=1,
                             activation=None)

    pred = tf.sigmoid(output)
    predictions = {"prob": tf.reshape(pred, [-1])}

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # ---------------------------------------------------------------------------

    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
        labels = tf.reshape(labels, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels))
        auc_val = tf.metrics.auc(labels, pred)
        eval_metric_ops = {"auc": auc_val}
    else:
        loss = None
        eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # do not forget this | important for batch_normalization
            optimizer = get_optimizer_instance(params["opt_algo"], params["learning_rate"])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    else:
        train_op = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = predictions
        export_outputs = export_outputs
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


def train_and_eval(model, train_data, eval_data, train_epochs, batch_size, epochs_per_eval):
    for n in range(train_epochs):
        t0 = time.time()
        print('\n' + '=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')

        tf.logging.info('\n' + '=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data))

        model.train(
            input_fn=lambda: input_fn(train_data, 1, 'train', batch_size),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)

        tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data, elapse_time(t0)))

        # every epochs_per_eval eval the model (use larger eval dataset)
        if (n + 1) % epochs_per_eval == 0:
            print('-' * 10 + ' evaluate at epoch {}'.format(n + 1) + '-' * 10)
            tf.logging.info('<EPOCH {}>: Start evaluate {}'.format(n + 1, eval_data))
            results = model.evaluate(
                input_fn=lambda: input_fn(eval_data, 1, 'eval', batch_size * 10),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                name=None)
            tf.logging.info(
                '<EPOCH {}>: Finish testing {}, take {} mins'.format(n + 1, eval_data, elapse_time(t0)))
            print('\n' + '-' * 30 + ' eval result at epoch {}'.format(n + 1) + '-' * 30 + '\n')
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))


def train(model, train_data, train_epochs, batch_size):
    for n in range(train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        t0 = time.time()
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data))
        model.train(
            input_fn=lambda: input_fn(train_data, 1, 'train', batch_size),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)
        tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data, elapse_time(t0)))


def train_and_eval_api(model, train_data, eval_data, train_epochs, batch_size):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_data, train_epochs, 'train', batch_size), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_data, 1, 'eval', batch_size))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def predict(model, pred_data, batch_size):
    tf.logging.info('=' * 30 + 'START PREDICTION' + '=' * 30)
    t0 = time.time()
    predictions = model.predict(
        input_fn=lambda: input_fn(pred_data, 1, 'pred', batch_size),
        predict_keys=None,
        hooks=None,
        checkpoint_path=None)  # defaults None to use latest_checkpoint

    tf.logging.info('=' * 30 + 'FINISH PREDICTION, TAKE {} mins'.format(elapse_time(t0)) + '=' * 30)
    return predictions  # dict {"prob": pred}


if __name__ == '__main__':
    pass
    # tf.logging.set_verbosity(tf.logging.INFO)
