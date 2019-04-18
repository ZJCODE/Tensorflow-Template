# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
import time
import shutil
import os
import re
import numpy as np
from .data import dataProcess
from .conf import Config
from .util import elapse_time

config = Config(base_dir='../conf')


def train(model, input_fn, train_data_path, train_epochs, batch_size):
    for n in range(train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        t0 = time.time()
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data_path))
        model.train(
            input_fn=lambda: input_fn(train_data_path, 1, 'train', batch_size),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)
        tf.logging.info(
            '<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data_path, elapse_time(t0)))


def train_and_eval(model, input_fn, train_data_path, eval_data_path, train_epochs, batch_size, epochs_per_eval=1):
    for n in range(train_epochs):
        t0 = time.time()
        print('\n' + '=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')

        tf.logging.info('\n' + '=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data_path))

        model.train(
            input_fn=lambda: input_fn(train_data_path, 1, 'train', batch_size),
            hooks=None,
            steps=None,
            max_steps=None,
            saving_listeners=None)

        tf.logging.info(
            '<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data_path, elapse_time(t0)))
        # every epochs_per_eval eval the model (use larger eval dataset)
        if (n + 1) % epochs_per_eval == 0:
            print('-' * 10 + ' evaluate at epoch {}'.format(n + 1) + '-' * 10)
            tf.logging.info('<EPOCH {}>: Start evaluate {}'.format(n + 1, eval_data_path))
            results = model.evaluate(
                input_fn=lambda: input_fn(eval_data_path, 1, 'eval', batch_size * 10),
                steps=None,  # Number of steps for which to evaluate model.
                hooks=None,
                checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                name=None)
            tf.logging.info(
                '<EPOCH {}>: Finish testing {}, take {} mins'.format(n + 1, eval_data_path, elapse_time(t0)))
            print('\n' + '-' * 30 + ' eval result at epoch {}'.format(n + 1) + '-' * 30 + '\n')
            # Display evaluation metrics
            for key in sorted(results):
                print('{}: {}'.format(key, results[key]))


def train_and_eval_api(model, input_fn, train_data_path, eval_data_path, train_epochs, batch_size):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(train_data_path, train_epochs, 'train', batch_size), max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(eval_data_path, 1, 'eval', batch_size))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def predict(model, input_fn, pred_data_path, batch_size):
    tf.logging.info('=' * 30 + 'START PREDICTION' + '=' * 30)
    t0 = time.time()
    predictions = model.predict(
        input_fn=lambda: input_fn(pred_data_path, 1, 'pred', batch_size),
        predict_keys=None,
        hooks=None,
        checkpoint_path=None)  # defaults None to use latest_checkpoint
    tf.logging.info('=' * 30 + 'FINISH PREDICTION, TAKE {} mins'.format(elapse_time(t0)) + '=' * 30)
    return predictions  # dict {"prob": pred}


def get_optimizer(opt_algo, learning_rate, loss, global_step):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adagrad':
        return tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    elif opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    elif opt_algo == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

    else:
        print("input error use adam for default")
        return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


def export_model(model, export_path, feature_spec):
    # feature_spec example
    # feature_spec = {'col1': tf.FixedLenFeature([], tf.float32, ),
    #                 'col2': tf.FixedLenFeature([], tf.float32),
    #                 'col3': tf.FixedLenFeature([], tf.float32),
    #                 'col4': tf.FixedLenFeature([], tf.string),
    #                 'col5': tf.FixedLenFeature([], tf.float32),
    #                 'col6': tf.FixedLenFeature([], tf.string),
    #                 'col7': tf.FixedLenFeature([], tf.string),
    #                 'col8': tf.FixedLenFeature([], tf.string)}

    def serving_input_receiver_fn():
        """An input receiver that expects a serialized tf.Example."""
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                               shape=[None],
                                               name='input_example_tensor')
        receiver_tensors = {'examples': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    model.export_savedmodel(export_path, serving_input_receiver_fn, strip_default_attrs=True)


def export_model_raw_input(model, export_path, feature_spec):
    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
    model.export_savedmodel(export_path, serving_input_receiver_fn, strip_default_attrs=True)


def load_model_predict(export_path, feature_dict, sess):
    # feature_dict example
    # feature_dict = {'user_id': "33569511",
    #                 'item_id': "1164006",
    #                 'item_id_sequence': "1164006,1164006",
    #                 'interact_time_gap_sequence': "10,1",
    #                 'interact_type_sequence': "d,-"}

    feature = {}
    for k, v in feature_dict.items():
        if isinstance(v, str):
            feature.update({k: _bytes_feature(v)})
        elif isinstance(v, int):
            feature.update({k: _int64_feature(v)})
        elif isinstance(v, float):
            feature.update({k: _float_feature(v)})
        else:
            print('wrong type')

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_path)
    predictor = tf.contrib.predictor.from_saved_model(export_path)
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    example = {'examples': [example.SerializeToString()]}
    output = predictor(example)
    return output

def load_model_raw_predict(raw_export_path, feature_dict, sess):
    # feature_dict example
    # feature_dict = {'user_id': "33569511",
    #                 'item_id': "1164006",
    #                 'item_id_sequence': "1164006,1164006",
    #                 'interact_time_gap_sequence': "10,1",
    #                 'interact_type_sequence': "d,-"}

    feature_input = {}
    for k, v in feature_dict.items():
        feature_input.update({k:[v]})
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], raw_export_path)
    predictor = tf.contrib.predictor.from_saved_model(raw_export_path)
    output = predictor(feature_input)
    return output

def get_embedding_weight(checkpoint_path, var_output_path, clean=False):
    tf.reset_default_graph()
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if clean:
            shutil.rmtree(var_output_path, ignore_errors=True)
            os.mkdir(var_output_path)

        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)

        var_list = tf.trainable_variables()
        embedding_var_list = [v for v in var_list if '_embedding' in v.name]

        while True:
            print('all embeeding var show as below : \n')
            for index, name in enumerate([v.name for v in embedding_var_list]):
                print(index + 1, ': ', name)

            index = int(input('\nchoose one to export [start from 1 , input -1 to exit] :  '))
            if index < 0:
                break
            embedding_var = embedding_var_list[index - 1]
            embedding_var_value = sess.run(embedding_var)
            embedding_name = re.findall('(\w+_embedding)', embedding_var.name)[0]
            np.savetxt(var_output_path + "/vocabulary_" + embedding_name, embedding_var_value,
                       fmt='%.4f',
                       delimiter=',',
                       encoding='utf8')
            print('\n==== write out {}\n'.format(embedding_name))


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    value = int(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    value = float(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = value.encode('utf8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
