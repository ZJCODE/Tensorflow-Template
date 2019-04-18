# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import tensorflow as tf
from utils.model import load_model_predict, load_model_raw_predict
from utils.conf import Config
import glob

config = Config(base_dir='../conf')


def test_load_model_predict():
    sess = tf.Session()
    feature_dict_example = {'col1': 2.0,
                            'col2': 3.0,
                            'col3': 5.0,
                            'col4': "a",
                            'col5': 3.0,
                            'col6': "r#w#k",
                            'col7': "f",
                            'col8': "e"}

    model_path_list = glob.glob(config.get_model_prop('model_export_dir') + '/*')
    export_path = max(model_path_list)

    p = load_model_predict(export_path=export_path,
                           feature_dict=feature_dict_example,
                           sess=sess)


    print(p)

def test_load_model_predict_raw():
    sess = tf.Session()
    # feature_dict_example = {'col1': 10.0,
    #                         'col2': 13.0,
    #                         'col3': 16.0,
    #                         'col4': "b",
    #                         'col5': 9.0,
    #                         'col6': "g#d#a",
    #                         'col7': "e",
    #                         'col8': "w"}

    feature_dict_example = {'col1': 2.0,
                            'col2': 3.0,
                            'col3': 5.0,
                            'col4': "a",
                            'col5': 3.0,
                            'col6': "r#w#k",
                            'col7': "e",
                            'col8': "e"}

    raw_model_path_list = glob.glob(config.get_model_prop('raw_model_export_dir') + '/*')
    raw_export_path = max(raw_model_path_list)

    p = load_model_raw_predict(raw_export_path=raw_export_path,
                               feature_dict=feature_dict_example,
                               sess=sess)

    print(p)


if __name__ == '__main__':
    test_load_model_predict()
    test_load_model_predict_raw()
