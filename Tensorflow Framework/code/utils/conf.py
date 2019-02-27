# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import yaml
import os

BASE_DIR = '../../conf'
DATA_CONF_FILE = 'data.yaml'
FEATURE_CONF_FILE = 'feature.yaml'
CROSS_FEATURE_CONF_FILE = 'cross_feature.yaml'
MODEL_CONF_FILE = 'model.yaml'


class Config(object):
    def __init__(self,
                 base_dir=BASE_DIR,
                 data_config_file=DATA_CONF_FILE,
                 feature_config_file=FEATURE_CONF_FILE,
                 model_config_file=MODEL_CONF_FILE):
        self.data_config_path = os.path.join(base_dir, data_config_file)
        self.model_config_path = os.path.join(base_dir, model_config_file)
        self.feature_config_path = os.path.join(base_dir, feature_config_file)

        self.data_config = self.read_data_conf()
        self.feature_config = self.read_feature_conf()
        self.model_config = self.read_model_conf()
        self.INPUT_RAW_FEATURE_NAMES = []
        self.INPUT_RAW_FEATURE_TYPES = []


        self.get_columns()
        self.get_data_scheme()

    def read_data_conf(self):
        with open(self.data_config_path) as f:
            data_conf = yaml.load(f)
            return data_conf

    def read_model_conf(self):
        with open(self.model_config_path) as f:
            mdoel_conf = yaml.load(f)
            return mdoel_conf

    def read_feature_conf(self):
        with open(self.feature_config_path) as f:
            feature_conf = yaml.load(f)
            for feature, conf in feature_conf.items():
                self._check_feature_conf(feature.lower(), **conf)
            return feature_conf

    def get_data_prop(self, key):
        return self.data_config[key]

    def get_model_prop(self, key):
        return self.model_config[key]

    def get_data_scheme(self):
        self.HEADER = []
        self.RECORD_DEFAULTS = []
        self.DTYPE = []
        self.SEQUENCE_COLS = []
        for k, v in self.data_config['data_schema'].items():
            self.HEADER.append(k)
            v_split = v.split('|')
            dtype = v_split[0]
            if len(v_split) == 2:
                self.SEQUENCE_COLS.append([k, v_split[1]])
            self.RECORD_DEFAULTS.append([self.type2instance(dtype)])
            self.DTYPE.append(dtype)
        self.FIELD_DELIM = self._sep_correct(self.data_config['field_delim'])

    def get_columns(self):
        # numerical [col]
        self.NUMERICAL_COLUMNS = [k for k, v in self.feature_config.items() if
                                  v['type'] == 'continuous' and v['transform'] == 'original']

        self.INPUT_RAW_FEATURE_NAMES += self.NUMERICAL_COLUMNS
        self.INPUT_RAW_FEATURE_TYPES += ['float' for x in self.NUMERICAL_COLUMNS]


        # bucket [[col,boundaries]...]
        self.BUCKET_COLUMNS = [[k, v['parameter']['boundaries']] for k, v in
                               self.feature_config.items() if
                               v['type'] == 'continuous' and v['transform'] == 'bucket']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.BUCKET_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['float' for x in self.BUCKET_COLUMNS]

        # hash category [one-hot] [[col , hash_size]...]
        self.HASH_CATEGORICAL_COLUMNS = [[k, v['parameter']['hash_size']] for k, v in
                                         self.feature_config.items() if
                                         v['type'] == 'category' and v['transform'] == 'hash_ont_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.HASH_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.HASH_CATEGORICAL_COLUMNS]

        # file category [one-hot] [[col,file_path]...]
        self.FILE_CATEGORICAL_COLUMNS = [[k, v['parameter']['file_path']] for k, v in
                                         self.feature_config.items() if
                                         v['type'] == 'category' and v['transform'] == 'file_one_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.FILE_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.FILE_CATEGORICAL_COLUMNS]

        # hash embedding [embedding] [[col , hash_size , embedding_size]...] | embedding_size == -1 means dim_auto
        self.HASH_EMBEDDING_COLUMNS = [[k, v['parameter']['hash_size'], v['parameter']['embedding_size']] for k, v in
                                       self.feature_config.items() if
                                       v['type'] == 'category' and v['transform'] == 'hash_embedding']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.HASH_EMBEDDING_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.HASH_EMBEDDING_COLUMNS]

        # file embedding [embedding] [[col , file_path,embedding_size]...]
        self.FILE_EMBEDDING_COLUMNS = [[k, v['parameter']['file_path'], v['parameter']['embedding_size']] for k, v in
                                       self.feature_config.items() if
                                       v['type'] == 'category' and v['transform'] == 'file_embedding']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.FILE_EMBEDDING_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.FILE_EMBEDDING_COLUMNS]

        # hash sequence category [sequence one-hot ] [[col , hash_size]...]
        self.HASH_SEQUENCE_CATEGORICAL_COLUMNS = [[k, v['parameter']['hash_size']] for k, v in
                                                  self.feature_config.items() if
                                                  v['type'] == 'sequence' and v['transform'] == 'hash_one_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.HASH_SEQUENCE_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.HASH_SEQUENCE_CATEGORICAL_COLUMNS]

        # file sequence category [sequence one-hot ] [[col,file_path]...]
        self.FILE_SEQUENCE_CATEGORICAL_COLUMNS = [[k, v['parameter']['file_path']] for k, v in
                                                  self.feature_config.items() if
                                                  v['type'] == 'sequence' and v['transform'] == 'file_one_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.FILE_SEQUENCE_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.FILE_SEQUENCE_CATEGORICAL_COLUMNS]

        # hash sequence embedding [sequence embedding ] [[col , hash_size,embedding_size]...]
        self.HASH_SEQUENCE_EMBEDDING_COLUMNS = [[k, v['parameter']['hash_size'], v['parameter']['embedding_size']] for
                                                k, v in
                                                self.feature_config.items() if
                                                v['type'] == 'sequence' and v['transform'] == 'hash_embedding']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.HASH_SEQUENCE_EMBEDDING_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.HASH_SEQUENCE_EMBEDDING_COLUMNS]

        # file sequence embedding [sequence embedding ] [[col,file_path,embedding_size]...]
        self.FILE_SEQUENCE_EMBEDDING_COLUMNS = [[k, v['parameter']['file_path'], v['parameter']['embedding_size']] for
                                                k, v in
                                                self.feature_config.items() if
                                                v['type'] == 'sequence' and v['transform'] == 'file_embedding']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.FILE_SEQUENCE_EMBEDDING_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.FILE_SEQUENCE_EMBEDDING_COLUMNS]

        # hash multi category [multi-hot] [[col , hash_size]...]
        self.HASH_MULTI_CATEGORICAL_COLUMNS = [[k, v['parameter']['hash_size']] for k, v in
                                               self.feature_config.items() if
                                               v['type'] == 'sequence' and v['transform'] == 'hash_multi_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.HASH_MULTI_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.HASH_MULTI_CATEGORICAL_COLUMNS]

        # file multi category [multi-hot] [[col,file_path]...]
        self.FILE_MULTI_CATEGORICAL_COLUMNS = [[k, v['parameter']['file_path']] for k, v in
                                               self.feature_config.items() if
                                               v['type'] == 'sequence' and v['transform'] == 'file_multi_hot']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.FILE_MULTI_CATEGORICAL_COLUMNS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.FILE_MULTI_CATEGORICAL_COLUMNS]

        # local vec  [[col , meta_path , vec_path]...]
        self.LOCAL_VECS = [[k, v['parameter']['meta_path'], v['parameter']['vec_path']] for k, v in
                           self.feature_config.items() if v['transform'] == 'local_vec']

        self.INPUT_RAW_FEATURE_NAMES += [x[0] for x in self.LOCAL_VECS]
        self.INPUT_RAW_FEATURE_TYPES += ['string' for x in self.LOCAL_VECS]

        # extend feature [[col,extend_file_path,extend_features_list,sep]]
        self.MAP_EXTEND_FEATURE = [
            [k.split('_[extend]')[0], v['parameter']['file_path'], v['parameter']['cols_name'],
             self._sep_correct(v['parameter']['extend_data_sep'])] for
            k, v in self.feature_config.items() if
            v['type'] in ('category', 'sequence') and v['transform'] == 'extend_feature']


    @staticmethod
    def _sep_correct(sep):
        if sep == '\\t':
            return '\t'
        else:
            return sep

    @staticmethod
    def _check_feature_conf(feature, **kwargs):
        type_ = kwargs["type"]
        trans = kwargs["transform"]
        para = kwargs["parameter"]
        if type_ is None:
            raise ValueError("Type are required in feature conf, "
                             "found empty value for feature `{}`".format(feature))
        assert type_ in {'category',
                         'continuous',
                         'sequence'}, (
            "Invalid type `{}` for feature `{}` in feature conf, "
            "must be 'category' or 'continuous' or 'sequence' ".format(type_, feature))
        # check transform and parameter
        if type_ == 'category':
            assert trans in {'file_one_hot',
                             'hash_ont_hot',
                             'file_embedding',
                             'hash_embedding',
                             'local_vec',
                             'extend_feature'}, (
                "Invalid transform `{}` for feature `{}` in feature conf".format(trans, feature))

        elif type_ == 'sequence':
            assert trans in {'file_one_hot',
                             'hash_ont_hot',
                             'file_embedding',
                             'hash_embedding',
                             'file_multi_hot',
                             'hash_multi_hot',
                             'extend_feature',
                             'local_vec'}, (
                "Invalid transform `{}` for feature `{}` in feature conf".format(trans, feature))

        else:
            pass

        if trans == 'file_one_hot':
            assert 'file_path' in para.keys(), ("should has file_path parameter for feature '{}' ".format(feature))
        elif trans == 'hash_ont_hot':
            assert 'hash_size' in para.keys(), ("should has hash_size parameter for feature '{}' ".format(feature))
        elif trans == 'file_embedding':
            assert 'file_path' in para.keys(), ("should has file_path parameter for feature '{}' ".format(feature))
            assert 'embedding_size' in para.keys(), (
                "should has embedding_size parameter for feature '{}' ".format(feature))
        elif trans == 'hash_embedding':
            assert 'hash_size' in para.keys(), ("should has hash_size parameter for feature '{}' ".format(feature))
            assert 'embedding_size' in para.keys(), (
                "should has embedding_size parameter for feature '{}' ".format(feature))
        elif trans == 'local_vec':
            assert 'meta_path' in para.keys(), ("should has meta_path parameter for feature '{}' ".format(feature))
            assert 'vec_path' in para.keys(), ("should has vec_path parameter for feature '{}' ".format(feature))
        elif trans == 'extend_feature':
            assert 'file_path' in para.keys(), ("should has file_path parameter for feature '{}' ".format(feature))
            assert 'cols_name' in para.keys(), ("should has cols_name parameter for feature '{}' ".format(feature))
            assert 'extend_data_sep' in para.keys(), (
                "should has extend_data_sep parameter for feature '{}' ".format(feature))
        elif trans == 'hash_multi_hot':
            assert 'hash_size' in para.keys(), ("should has hash_size parameter for feature '{}' ".format(feature))
        elif trans == 'file_multi_hot':
            assert 'file_path' in para.keys(), ("should has file_path parameter for feature '{}' ".format(feature))
        else:
            pass

    @staticmethod
    def type2instance(dtype):
        if dtype == 'string':
            return ''
        elif dtype == 'float':
            return 0.0
        elif dtype == 'int':
            return 0
        else:
            return ''


if __name__ == '__main__':
    config = Config()

    import json

    # print(json.dumps(config.data_config))
    # print(json.dumps(config.model_config))
    # print(json.dumps(config.train_config))
    # print(json.dumps(config.feature_config))

    print('# numerical [col]')
    print(config.NUMERICAL_COLUMNS)
    print('# bucket [[col,boundaries]...]')
    print(config.BUCKET_COLUMNS)
    print('# hash category [one-hot] [[col , hash_size]...]')
    print(config.HASH_CATEGORICAL_COLUMNS)
    print('# file category [one-hot] [[col,file_path]...]')
    print(config.FILE_CATEGORICAL_COLUMNS)
    print('# hash embedding [embedding] [[col , hash_size , embedding_size]...] | embedding_size == -1 means dim_auto')
    print(config.HASH_EMBEDDING_COLUMNS)
    print('# file embedding [embedding] [[col , file_path,embedding_size]...]')
    print(config.FILE_EMBEDDING_COLUMNS)
    print('# hash sequence category [sequence one-hot ] [[col , hash_size]...]')
    print(config.HASH_SEQUENCE_CATEGORICAL_COLUMNS)
    print(' # file sequence category [sequence one-hot ] [[col,file_path]...]')
    print(config.FILE_SEQUENCE_CATEGORICAL_COLUMNS)
    print('# hash sequence embedding [sequence embedding ] [[col , hash_size,embedding_size]...]')
    print(config.HASH_SEQUENCE_EMBEDDING_COLUMNS)
    print('# file sequence embedding [sequence embedding ] [[col,file_path,embedding_size]...]')
    print(config.FILE_SEQUENCE_EMBEDDING_COLUMNS)
    print('# hash multi category [multi-hot] [[col , hash_size]...]')
    print(config.HASH_MULTI_CATEGORICAL_COLUMNS)
    print('# file multi category [multi-hot] [[col,file_path]...]')
    print(config.FILE_MULTI_CATEGORICAL_COLUMNS)
    print('# local vec  [[col, meta_path , vec_path]...]')
    print(config.LOCAL_VECS)
    print('# extend feature [[col,extend_file_path,extend_features_list,sep]]')
    print(config.MAP_EXTEND_FEATURE)
    print('# sequence cols [[col,sep],...]')
    print(config.SEQUENCE_COLS)
    print('# header')
    print(config.HEADER)
    print('# dtype')
    print(config.DTYPE)
    print('# record_default')
    print(config.RECORD_DEFAULTS)
    print('# label_col')
    print(config.get_data_prop('label_col'))
    print('# raw input feature names')
    print(config.INPUT_RAW_FEATURE_NAMES)
    print('# raw input feature types')
    print(config.INPUT_RAW_FEATURE_TYPES)
