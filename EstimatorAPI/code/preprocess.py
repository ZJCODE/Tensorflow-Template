# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

@author: zhangjun
"""

import time
import config
from utils.feature_stat import FeatureStat

if config.LOCAL_TRAIN:
    train_data_path = '../local_data/train'
    ids_path = '../local_data/ids/ids_'
else:
    train_data_path = '../data/train'
    ids_path = '../data/ids/ids_'


# basic

def get_ids_from_train_data():
    params = {'cols': config.HEADER,
              'dtypes': ['string' if isinstance(x[0], str) else 'num' for x in config.RECORD_DEFAULTS],
              'sep': config.FIELD_DELIM}
    select_cols = config.GET_IDS_COLUMNS
    sequence_cols = dict(
        zip(config.SEVERAL_VALUES_COLUMNS,
            [config.SEVERAL_VALUES_COLUMNS_SEP] * len(config.SEVERAL_VALUES_COLUMNS)))

    fs = FeatureStat(file_path=train_data_path,
                     pool_num=config.NUM_PARALLEL,
                     params=params,
                     select_cols=select_cols,
                     sequence_cols=sequence_cols)

    print(fs.string_stat)

    for col in config.GET_IDS_COLUMNS:
        print("write : ", col)
        string_col_file = open(ids_path + col, 'w')
        for data in fs.string_stat[col].keys():
            string_col_file.write(data + '\n')
        string_col_file.close()


if __name__ == '__main__':
    get_ids_from_train_data()
