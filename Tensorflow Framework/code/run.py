# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import os
import tensorflow as tf
import shutil
from utils.conf import Config
from input_fn import input_fn
from model_fn import model_fn
from utils.model import train_and_eval, export_model, export_model_raw_input
from utils.util import prase_feature_placehold, prase_feature
from utils.util import get_free_gpus

config = Config(base_dir='../conf')


def main():
    if not config.get_model_prop('keep_train'):
        shutil.rmtree(config.get_model_prop('model_check_dir'), ignore_errors=True)
        print('Remove model directory: {}'.format(config.get_model_prop('model_check_dir')))

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_free_gpus()[-1]
    except:
        pass

    model_params = {'opt_algo': config.get_model_prop('opt_algo'),
                    'learning_rate': config.get_model_prop('learning_rate')}

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                    device_count={'CPU': config.get_model_prop('num_parallel')})

    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        log_step_count_steps=config.get_model_prop('log_steps'))

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=config.get_model_prop('model_check_dir'),
                                   params=model_params,
                                   config=estimator_config)

    train_and_eval(model=model,
                   input_fn=input_fn,
                   train_data_path=config.get_data_prop('train_data_path'),
                   eval_data_path=config.get_data_prop('eval_data_path'),
                   train_epochs=config.get_model_prop('train_epochs'),
                   batch_size=config.get_model_prop('batch_size'),
                   epochs_per_eval=1)

    # feature_spec = {'col1': tf.FixedLenFeature([], tf.float32),
    #                 'col2': tf.FixedLenFeature([], tf.float32),
    #                 'col3': tf.FixedLenFeature([], tf.float32),
    #                 'col4': tf.FixedLenFeature([], tf.string),
    #                 'col5': tf.FixedLenFeature([], tf.float32),
    #                 'col6': tf.FixedLenFeature([], tf.string),
    #                 'col7': tf.FixedLenFeature([], tf.string),
    #                 'col8': tf.FixedLenFeature([], tf.string)}

    feature_spec = {}

    for name, dtype in zip(config.HEADER, config.DTYPE):
        if name in config.get_data_prop('input_except'):
            pass
        else:
            feature_spec.update({name: prase_feature(dtype)})

    print(feature_spec)

    export_model(model=model,
                 export_path=config.get_model_prop('model_export_dir'),
                 feature_spec=feature_spec)

    # feature_raw_spec = {'col1': tf.placeholder(tf.float32, [None]),
    #                     'col2': tf.placeholder(tf.float32, [None]),
    #                     'col3': tf.placeholder(tf.float32, [None]),
    #                     'col4': tf.placeholder(tf.string, [None]),
    #                     'col5': tf.placeholder(tf.float32, [None]),
    #                     'col6': tf.placeholder(tf.string, [None]),
    #                     'col7': tf.placeholder(tf.string, [None]),
    #                     'col8': tf.placeholder(tf.string, [None])}
    feature_raw_spec = {}
    for name, dtype in zip(config.HEADER, config.DTYPE):
        feature_raw_spec.update({name: prase_feature_placehold(dtype)})

    print(feature_raw_spec)

    export_model_raw_input(model=model,
                           export_path=config.get_model_prop('raw_model_export_dir'),
                           feature_spec=feature_raw_spec)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
