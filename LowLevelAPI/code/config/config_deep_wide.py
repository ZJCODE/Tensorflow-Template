# -*- coding: utf-8 -*-
"""
Created on 2018.8.7

@author: zhangjun
"""

import multiprocessing

MULTI_THREADING = True

NUM_PARALLEL = multiprocessing.cpu_count() if MULTI_THREADING else 1


########################################################################
# data header info 
# [[col,[data default]],[],...]
########################################################################
# HEADER_INFO = []

HEADER_INFO = [['label',[0.0]],
               ['id',['']],
               ['num1',[0.0]],
               ['num2',[0.0]],
               ['cate1',['']],
               ['cate2',['']],
               ['seq',['']]
                   ]

HEADER,RECORD_DEFAULTS = zip(*HEADER_INFO)

########################################################################
# all numerical columns and will get summary for those columns
# [col,...]
########################################################################

NUMERICAL_COLUMNS = ['num1','num2']


########################################################################
# do scale to select numerical columns
# [col,...]
########################################################################

NUMERICAL_COLUMNS_FOR_SCALE = ['num1','num1']


SCALE_TYPE = 'norm'

########################################################################
# do numercial feature transform such as log,square,sqrt 
# [[col,func],[],...]
########################################################################

NUMERICAL_COLUMNS_FOR_TRANSFORM = [['num1','log'],['num2','log']]



########################################################################
# will do reindex for category columns and get summary [unique cnt]
# [col,...]
########################################################################
CATEGORY_COLUMNS = ['id','cate1','cate2']

SHARE_EMBEDDING_COLUMNS = []

########################################################################
# onehot those columns based on reindex value
# [col,...]
########################################################################
CATEGORY_COLUMNS_FOR_ONEHOT = ['cate1','cate2']


########################################################################
# embedding those columns based on reindex value
# [[col,embedding_size],[],...]
########################################################################

CATEGORY_COLUMNS_FOR_EMBEDDING = [['id',10]]


########################################################################
# add feature cross 
# [[col1,col2],[],...]
########################################################################

CATEGORY_COLUMNS_FOR_CROSS = [['cate1','cate2']]

########################################################################
# all sequence columns
# [col1,...]
########################################################################

SEQUENCE_COLUMNS = ['seq']
