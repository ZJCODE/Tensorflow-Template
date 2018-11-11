# -*- coding: utf-8 -*-
"""
Created on 2018.8.10

@author: zhangjun
"""

import multiprocessing

MULTI_THREADING = True

NUM_PARALLEL = multiprocessing.cpu_count() if MULTI_THREADING else 1


########################################################################
# data header info 
# [[col,[data default]],[],...]
########################################################################

HEADER_INFO = [['center_word',['']],['target_word',['']]]
HEADER,RECORD_DEFAULTS = zip(*HEADER_INFO)

########################################################################
# all numerical columns and will get summary for those columns
# [col,...]
########################################################################

NUMERICAL_COLUMNS = []

########################################################################
# do scale to select numerical columns
# [col,...]
########################################################################

NUMERICAL_COLUMNS_FOR_SCALE = []

SCALE_TYPE = 'norm' # norm or maxmin

########################################################################
# do numercial feature transform such as log,square,sqrt 
# [[col,func],[],...]
########################################################################

NUMERICAL_COLUMNS_FOR_TRANSFORM = []

########################################################################
# will do reindex for category columns and get summary [unique cnt]
# [col,...]
########################################################################

# prepear a files contains all words for train 
CATEGORY_COLUMNS = ['center_word','target_word']

SHARE_EMBEDDING_COLUMNS = ['words']

########################################################################
# onehot those columns based on reindex value
# [col,...]
########################################################################

CATEGORY_COLUMNS_FOR_ONEHOT = []


########################################################################
# embedding those columns based on reindex value
# [[col,embedding_size],[],...]
########################################################################

CATEGORY_COLUMNS_FOR_EMBEDDING = []


########################################################################
# add feature cross 
# [[col1,col2],[],...]
########################################################################

CATEGORY_COLUMNS_FOR_CROSS = []


SEQUENCE_COLUMNS = []

