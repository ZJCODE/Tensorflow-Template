# -*- coding: utf-8 -*-
"""
Created on 2018.10.10

@author: zhangjun
"""

import tensorflow as tf
import config


def export_model(model, export_dir):
    features = {'col1': tf.placeholder(tf.float32, shape=[None]),
                'col2': tf.placeholder(tf.float32, shape=[None]),
                'col3': tf.placeholder(tf.float32, shape=[None]),
                'col4': tf.placeholder(tf.string, shape=[None]),
                'col5': tf.placeholder(tf.float32, shape=[None]),
                'col6': tf.placeholder(tf.string, shape=[None]),
                'col7': tf.placeholder(tf.string, shape=[None]),
                'col8': tf.placeholder(tf.string, shape=[None])}

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)

    model.export_savedmodel(export_dir, serving_input_receiver_fn, strip_default_attrs=True)
