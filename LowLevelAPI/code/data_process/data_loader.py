# -*- coding: utf-8 -*-
"""
Created on 2018.8.8

@author: zhangjun
"""

import sys
sys.path.append("..")

import numpy as np
import tensorflow as tf
import config.config as config
import utils.utils as utils
import json

class Data(object):

    def __init__(self,
                 data_path,
                 data_name,
                 batch_size,
                 is_train_period,
                 sep,
                 low_freq_threshold = 0):

        self.data_path = data_path
        self.data_name = data_name
        self.batch_size = batch_size
        self.is_train_period = is_train_period
        self.sep = sep
        self.low_freq_threshold = low_freq_threshold

        self.dataset = None
        self.vocabulary_dicts = []
        self.sequence_vocabulary_dicts = []

        self.category_summary_dict = None #  key(cate name) : value(cate size)
        self.numerical_summary_dict = None  #  statistics info for numerical columns
        self.sequence_summary_dict = None

        self.numerical_col = config.NUMERICAL_COLUMNS
        self.onehot_col = []
        self.embedding_col = list(zip(*config.CATEGORY_COLUMNS_FOR_EMBEDDING))[0] if len(config.CATEGORY_COLUMNS_FOR_EMBEDDING) > 0 else []
        self.embedding_col_dict = dict(config.CATEGORY_COLUMNS_FOR_EMBEDDING)
        self.sequence_col = config.SEQUENCE_COLUMNS

    def read_dataset(self,sequence_columns = [],sequence_sep = ',',sep='\001'):
        """
        read data by using TextLineDataset and transform to dataset
        """
        self.dataset = tf.data.TextLineDataset(self.data_path + self.data_name)

        if self.is_train_period:        
            # in training period we will shuffle data     
            self.dataset = self.dataset.shuffle(5 * self.batch_size)

        # do csv parse / if the file we input is not a csv file ,we need to rewrite codes below
        self.dataset = self.dataset.map(lambda csv_row: utils.parse_csv_row(csv_row,config.HEADER,config.RECORD_DEFAULTS,sep),
                                        num_parallel_calls = config.NUM_PARALLEL)
        # add sequence length if there are some sequence columns 
        if len(sequence_columns) > 0:
            for col in sequence_columns:
                self.dataset = self.dataset.map(lambda row : utils.get_sequence_length(row,col,sequence_sep))

        # batch
        self.dataset = self.dataset.batch(self.batch_size)

    def _get_vocabulary_dicts(self,category_columns,share,low_freq_threshold):
        """
        read vocabulary_* files which is generated by data_summary process 
        generate vocabulary_dicts for category reindex 
        """

        # specific for word2vec start
        if len(share) > 0 :
            category_columns = share # just one 
        # specific for word2vec end

        for i,col in enumerate(category_columns):
            vocabularys = []
            freq_dict = {}
            with open(self.data_path + 'vocabulary_' + col,'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split(',')
                    vocabulary = line_split[0].encode('utf8')
                    count = int(line_split[1])
                    freq_dict[vocabulary] = count
                    vocabularys.append(vocabulary)
            vocabulary_dict = dict(zip(vocabularys,np.arange(1,len(vocabularys)+1).astype(np.int32)))
            for k,v in freq_dict.items():
                if v < low_freq_threshold:
                    vocabulary_dict[k] = np.array([0]).astype(np.int32)[0]
            with open(self.data_path + 'vocabulary_' + col + '_reindex','w') as f:
                for k,v in vocabulary_dict.items():
                    f.write(str(k.decode()) + ',' + str(v) + '\n')
            self.vocabulary_dicts.append(vocabulary_dict)  # keep the index 0 for oov
        

    def _get_category_summary_dict(self):
        """
        read category_summary files which is generated by data_summary process 
        generate category_summary_dict for later usage like onehot and embedding
        """
        with open(self.data_path + 'category_summary') as f:
            self.category_summary_dict = json.loads(f.readline())


    # no new col name
    def _category_reindex(self,category_columns,share): # for embedding or onehot
        """
        do category reindex 
        so we do not need to do hash on category columns
        this can avoid using too much memory and confliction
        """
        for i,col in enumerate(category_columns):

            # specific for word2vec start
            if len(share) > 0:            
                self.dataset = self.dataset.map(lambda row : utils.category_reindex(row,col,self.vocabulary_dicts[0]),   
                                                num_parallel_calls = config.NUM_PARALLEL)
            # specific for word2vec end

            else:
                self.dataset = self.dataset.map(lambda row : utils.category_reindex(row,col,self.vocabulary_dicts[i]),   
                                                num_parallel_calls = config.NUM_PARALLEL)


    def _get_sequence_summary_dict(self):
        """
        read category_summary files which is generated by data_summary process 
        generate category_summary_dict for later usage like onehot and embedding
        """
        with open(self.data_path + 'sequence_summary') as f:
            self.sequence_summary_dict = json.loads(f.readline())


    def _get_sequence_vocabulary_dicts(self,sequence_columns,low_freq_threshold):
        """
        read vocabulary_* files which is generated by data_summary process 
        generate vocabulary_dicts for category reindex 
        """

        for i,col in enumerate(sequence_columns):
            vocabularys = []
            freq_dict = {}
            with open(self.data_path + 'sequence_vocabulary_' + col,'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split(',')
                    vocabulary = line_split[0].encode('utf8')
                    count = int(line_split[1])
                    freq_dict[vocabulary] = count
                    vocabularys.append(vocabulary)
            vocabulary_dict = dict(zip(vocabularys,np.arange(1,len(vocabularys)+1).astype(np.int32)))
            for k,v in freq_dict.items():
                if v < low_freq_threshold:
                    vocabulary_dict[k] = np.array([0]).astype(np.int32)[0]
            with open(self.data_path + 'sequence_vocabulary_' + col + '_reindex','w') as f:
                for k,v in vocabulary_dict.items():
                    f.write(str(k.decode()) + ',' + str(v) + '\n')
            self.sequence_vocabulary_dicts.append(vocabulary_dict)  # keep the index 0 for oov
        

    def _sequence2dense(self,sequence_columns,sequence_sep):
        for col in sequence_columns:
            self.dataset = self.dataset.map(lambda row:utils.string2dense(row,col,sequence_sep))

    # no new col name
    def _sequence_reindex(self,sequence_columns): # for embedding or onehot
        """
        do category reindex 
        so we do not need to do hash on category columns
        this can avoid using too much memory and confliction
        """
        for i,col in enumerate(sequence_columns):
            self.dataset = self.dataset.map(lambda row : utils.category_seq_reindex(row,col,self.sequence_vocabulary_dicts[i],self.batch_size),   
                                                num_parallel_calls = config.NUM_PARALLEL)


    # new col name
    def _category_onehot(self,category_columns_for_onehot):
        """
        
        """
        for col in category_columns_for_onehot:
            self.onehot_col.append(col + "_onehot")
            self.dataset = self.dataset.map(lambda row : utils.onehot(row,col,self.category_summary_dict[col]),
                                            num_parallel_calls = config.NUM_PARALLEL)


    def _get_numerical_summary_dict(self):
        with open(self.data_path + 'numerical_summary') as f:
            self.numerical_summary_dict = json.loads(f.readline())

    # new col name
    def _numerical_transform(self,numercial_columns_for_transform):
        for col,way in numercial_columns_for_transform:
            self.numerical_col.append(col + '_' + way)
            self.dataset = self.dataset.map(lambda row : utils.transform(row,col,way),
                                            num_parallel_calls = config.NUM_PARALLEL)
    # new col name
    def _numerical_scale(self,numerical_columns_for_scale,type='norm'):
        for col in numerical_columns_for_scale:    
            if type == 'norm':
                self.numerical_col.append(col+"_norm_scale")
                self.dataset = self.dataset.map(lambda row: utils.norm_scale(row,col,self.numerical_summary_dict),
                                                num_parallel_calls = config.NUM_PARALLEL)
            elif type == 'maxmin':
                self.numerical_col.append(col+"_maxmin_scale")
                self.dataset = self.dataset.map(lambda row: utils.max_min_scale(row,col,self.numerical_summary_dict),
                                                num_parallel_calls = config.NUM_PARALLEL)
            else:
                pass

    # new col name
    def _feature_cross(self,category_columns_for_cross):
        for col1,col2 in category_columns_for_cross:
            self.onehot_col.append('cross_' + col1 + "_" + col2)
            self.dataset = self.dataset.map(lambda row: utils.cross_feature(row,col1,col2,self.category_summary_dict[col1] * self.category_summary_dict[col2]))

    # more efficient
    def _final(self):
        self.dataset = self.dataset.prefetch(1)
  
    def get_dataset(self):
        self.read_dataset(sequence_columns = config.SEQUENCE_COLUMNS,sequence_sep = ',',sep = self.sep)

        if len(config.CATEGORY_COLUMNS) > 0:
            self._get_vocabulary_dicts(config.CATEGORY_COLUMNS,config.SHARE_EMBEDDING_COLUMNS,self.low_freq_threshold)
            self._get_category_summary_dict() 

        if len(config.SEQUENCE_COLUMNS) > 0:
            self._get_sequence_vocabulary_dicts(config.SEQUENCE_COLUMNS,self.low_freq_threshold)
            self._get_sequence_summary_dict()
            self._sequence2dense(config.SEQUENCE_COLUMNS,',')
            self._sequence_reindex(config.SEQUENCE_COLUMNS)

        if len(config.CATEGORY_COLUMNS_FOR_CROSS) > 0:
            self._feature_cross(config.CATEGORY_COLUMNS_FOR_CROSS) # must before _category_reindex

        if len(config.CATEGORY_COLUMNS) > 0:
            self._category_reindex(config.CATEGORY_COLUMNS,config.SHARE_EMBEDDING_COLUMNS)

        if len(config.CATEGORY_COLUMNS_FOR_ONEHOT) > 0:
            self._category_onehot(config.CATEGORY_COLUMNS_FOR_ONEHOT)

        if len(config.NUMERICAL_COLUMNS) > 0:
            self._get_numerical_summary_dict()

        if len(config.NUMERICAL_COLUMNS_FOR_TRANSFORM) > 0 :
            self._numerical_transform(config.NUMERICAL_COLUMNS_FOR_TRANSFORM)

        if len(config.NUMERICAL_COLUMNS_FOR_SCALE) > 0:
            self._numerical_scale(config.NUMERICAL_COLUMNS_FOR_SCALE,config.SCALE_TYPE)

        self._final()



if __name__ == '__main__':
    data = Data(data_path = '../../data/',
                 data_name = 'train_data',
                 batch_size = 3,
                 is_train_period = False,
                 sep = ' ',
                 low_freq_threshold = 0)

    data.get_dataset()

    dataset = data.dataset

    i = dataset.make_one_shot_iterator()
    gen = i.get_next()

    sess = tf.Session()

    print(sess.run(gen))
    print(sess.run(gen))








