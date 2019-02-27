# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

from utils.data import dataStat
from utils.conf import Config

config = Config(base_dir='../conf')


def get_ids_from_train_data():
    params = {'cols': config.HEADER,
              'dtypes': ['string' if isinstance(x[0], str) else 'num' for x in config.RECORD_DEFAULTS],
              'sep': config.FIELD_DELIM,
              'select_cols': config.get_data_prop('vocabulary_col'),
              'sequence_cols': dict(config.SEQUENCE_COLS)}

    fs = dataStat(file_path=config.get_data_prop('train_data_path'),
                  pool_num=config.get_model_prop('num_parallel'),
                  params=params)

    for col in config.get_data_prop('vocabulary_col'):
        print("write : ", col)
        string_col_file = open(config.get_data_prop('vocabulary_path') + '/vocabulary_' + col, 'w', encoding='utf8')
        for data in fs.string_stat[col].keys():
            string_col_file.write(data + '\n')
        string_col_file.close()


def get_ids_from_map_data():
    for _, file_path, extend_features_list, sep in config.MAP_EXTEND_FEATURE:
        params = {'cols': extend_features_list,
                  'dtypes': ['string'] * len(extend_features_list),
                  'sep': sep,
                  'select_cols': extend_features_list,
                  'sequence_cols': {}}

        fs = dataStat(file_path=file_path,
                      pool_num=1,
                      params=params,
                      is_path=False)

        for col in extend_features_list:
            print("write : ", col)
            string_col_file = open(config.get_data_prop('vocabulary_path') + '/vocabulary_' + col, 'w', encoding='utf8')
            for data in fs.string_stat[col].keys():
                string_col_file.write(data + '\n')
            string_col_file.close()


if __name__ == '__main__':
    get_ids_from_train_data()
    get_ids_from_map_data()
