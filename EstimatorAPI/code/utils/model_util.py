# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

@author: zhangjun
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six


def add_layer_summary(value, tag):
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def get_optimizer_instance(opt, learning_rate=None):
    """Returns an optimizer instance.
    Supports the following types for the given `opt`:
        * An `Optimizer` instance string: Returns the given `opt`.
        * A supported string: Creates an `Optimizer` subclass with the given `learning_rate`.
      Supported strings:
        * 'Adagrad': Returns an `AdagradOptimizer`.
        * 'Adam': Returns an `AdamOptimizer`.
        * 'Ftrl': Returns an `FtrlOptimizer`.
        * 'RMSProp': Returns an `RMSPropOptimizer`.
        * 'SGD': Returns a `GradientDescentOptimizer`.
    Args:
      opt: An `Optimizer` instance, or supported string, as discussed above.
      learning_rate: A float. Only used if `opt` is a supported string.
    Returns:
      An `Optimizer` instance.
    Raises:
      ValueError: If `opt` is an unsupported string.
      ValueError: If `opt` is a supported string but `learning_rate` was not specified.
      ValueError: If `opt` is none of the above types.
    """
    # Methods related to optimizers used in canned_estimators."""
    _OPTIMIZER_CLS_NAMES = {
        'Adagrad': tf.train.AdagradOptimizer,
        'Adam': tf.train.AdamOptimizer,
        'Ftrl': tf.train.FtrlOptimizer,
        'RMSProp': tf.train.RMSPropOptimizer,
        'SGD': tf.train.GradientDescentOptimizer
    }
    if isinstance(opt, six.string_types):
        if opt in six.iterkeys(_OPTIMIZER_CLS_NAMES):
            if learning_rate is None:
                raise ValueError('learning_rate must be specified when opt is supported string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate)
        else:
            try:
                opt = eval(opt)  # eval('tf.nn.relu') tf.nn.relu
                if not isinstance(opt, tf.train.Optimizer):
                    raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
                return opt
            except (AttributeError, NameError):
                raise ValueError('Unsupported optimizer option: `{}`. '
                                 'Supported names are: {} or an `Optimizer` instance.'.format(
                    opt, tuple(sorted(six.iterkeys(_OPTIMIZER_CLS_NAMES)))))

