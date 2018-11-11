# -*- coding: utf-8 -*-
"""
Created on 2018.9.28

@author: zhangjun
"""

import multiprocessing

MULTI_THREADING = True
LOCAL_TRAIN = True

NUM_PARALLEL = multiprocessing.cpu_count() if MULTI_THREADING else 1

# data header info [[col,[data default]],[],...]
HEADER_INFO = [['col1', [0.0]],
               ['col2', [0.0]],
               ['col3', [0.0]],
               ['col4', ['']],
               ['col5', [0.0]],
               ['col6', ['']],
               ['col7', ['']],
               ['col8', ['']],
               ['label', [0.0]]]

HEADER, RECORD_DEFAULTS = zip(*HEADER_INFO)

FIELD_DELIM = ','

if LOCAL_TRAIN:
    DATA_PATH = '../local_data'
    RSYNC_MODEL_DIR = '../model_files/temp_rsync'
else:
    DATA_PATH = '../data'
    RSYNC_MODEL_DIR = ''

# ------------------- Denote -------------------

# several values [col]
SEVERAL_VALUES_COLUMNS = ['col6', 'col8']
# several values sep
SEVERAL_VALUES_COLUMNS_SEP = '#'
# ids [col]
GET_IDS_COLUMNS = ['col4', 'col6', 'col7', 'col8']

# ------------------- Extend -------------------

# extend feature [[col,extend_file_path,extend_features_list]]
MAP_EXTEND_FEATURE = [
    ['col7',
     DATA_PATH + '/map/extend_data_sample',
     ['extend_1', 'extend_2']
     ]
]

EXTEND_FEATURE_SEP = ' '

# example : new added columns name : col7_extend_1 , col7_extend_2

# ------------------- Feature -------------------


# numerical [col]
NUMERICAL_COLUMNS = ['col1', 'col2', 'col3', 'col5']

# bucket [[col,boundaries]...]
BUCKET_COLUMNS = [['col1', [0, 5, 10]]]

# hash category [one-hot] [[col , hash_size]...]
HASH_CATEGORICAL_COLUMNS = [['col4', 10], ['col7_extend_1', 10]]
# file category [one-hot] [[col,file_path]...]
FILE_CATEGORICAL_COLUMNS = [['col4', DATA_PATH + '/ids/ids_col4'], ['col7', DATA_PATH + '/ids/ids_col7']]

# hash embedding [embedding] [[col , hash_size , embedding_size]...] | embedding_size == -1 means dim_auto
HASH_EMBEDDING_COLUMNS = [['col4', 10, -1]]
# file embedding [embedding] [[col , file_path,embedding_size]...]
FILE_EMBEDDING_COLUMNS = [['col4', DATA_PATH + '/ids/ids_col4', 5]]

# SEQUENCE_CATEGORICAL_COLUMNS and MULTI_CATEGORICAL_COLUMNS must be one of SEVERAL_VALUES_COLUMNS

# hash sequence category [sequence one-hot ] [[col , hash_size]...]
HASH_SEQUENCE_CATEGORICAL_COLUMNS = [['col6', 10]]
# file sequence category [sequence one-hot ] [[col,file_path]...]
FILE_SEQUENCE_CATEGORICAL_COLUMNS = [['col6', DATA_PATH + '/ids/ids_col6'],
                                     ['col8', DATA_PATH + '/ids/ids_col8']]

# hash sequence embedding [sequence embedding ] [[col , hash_size,embedding_size]...]
HASH_SEQUENCE_EMBEDDING_COLUMNS = [['col6', 10, 5]]
# file sequence embedding [sequence embedding ] [[col,file_path,embedding_size]...]
FILE_SEQUENCE_EMBEDDING_COLUMNS = [['col6', DATA_PATH + '/ids/ids_col6', 5]]

# hash multi category [multi-hot] [[col , hash_size]...]
HASH_MULTI_CATEGORICAL_COLUMNS = [['col8', 10]]
# file multi category [multi-hot] [[col,file_path]...]
FILE_MULTI_CATEGORICAL_COLUMNS = [['col8', DATA_PATH + '/ids/ids_col8'], ['col6', DATA_PATH + '/ids/ids_col6']]

# local vec  [[col ,vec_dim, meta_path , vec_path]...]
LOCAL_VECS = [
    ['col4', 3, DATA_PATH + '/ids_vecs/meta_sample', DATA_PATH + '/ids_vecs/vec_sample'],
    ['col7', 3, DATA_PATH + '/ids_vecs/meta_sample', DATA_PATH + '/ids_vecs/vec_sample']]

# label
LABEL_COLUMN = 'label'

# ------------------- Model -------------------


MODEL_CHECK_DIR = '../model_files/model_check'
MODEL_EXPORT_DIR = '../model_files/export'

TRAIN_DATA_PATH = DATA_PATH + '/train'
EVAL_DATA_PATH = DATA_PATH + '/eval'
PREDICT_DATA_PATH = DATA_PATH + '/pred'
