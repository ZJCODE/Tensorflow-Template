# -*- coding: utf-8 -*-
"""
Created on 2018.8.10

@author: zhangjun
"""

import sys
sys.path.append("..")
import time
import tensorflow as tf 
import model_tools.model_base as model_base
import model_tools.model_blocks as model_blocks
import numpy as np

class word2vec_model(model_base.model):

    def __init__(self,sess,batch_size,learning_rate,
                 opt_algo,epochs,regular_rate,dropout_rate,skip_step,
                 num_sampled,embed_size):

        super().__init__(sess,batch_size,learning_rate,
                         opt_algo,epochs,regular_rate,dropout_rate,skip_step)

        self.checkpoint_dir = 'checkpoint/word2vec/model'
        self.num_sampled = num_sampled
        self.embed_size = embed_size

    ######################
    # modify star *      #
    ######################

    # def get_data(self):
    #     pass

    ######################
    # modify star **     #
    ######################

    def get_label(self):
        # set label as target_word [positive]
        # negative label by using negative samping
        self.target_word = tf.reshape(self.data_generator['target_word'],[-1,1])

    ######################
    # modify star *****  #
    ######################

    def inference(self):
        self.vocab_size = self.data_info.category_summary_dict['words'] +1 # add one , because we level index 0 for oov

        # shape [vocab_size,embedding_size]
        self.embed_matrix = tf.Variable(tf.random_uniform(
                                          [self.vocab_size , self.embed_size]))
        # shape [vocab_size,embedding_size]
        self.nce_weight = tf.Variable(tf.truncated_normal(
                                        [self.vocab_size, self.embed_size],
                                        stddev=1.0 / (self.embed_size ** 0.5)))

        self.nce_bias = tf.Variable(tf.zeros([self.vocab_size]))
        # get embedding vec for center word
        self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.data_generator['center_word'])

    ######################
    # modify star ***    #
    ######################

    def model_loss(self):
        # build nec loss
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight, 
                                    biases=self.nce_bias, 
                                    labels= self.target_word, 
                                    inputs=self.embed, 
                                    num_sampled=self.num_sampled, 
                                    num_classes=self.vocab_size))

    ######################
    # modify star ***    #
    ######################

    # def model_optimizer(self):
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def eval(self):
    #     pass

    ######################
    # modify star *      #
    ######################

    # def bulid(self):
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def train_one_epoch(self,init,epoch):
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def eval_once(self,init,epoch):
    #     pass

    ######################
    # modify star **     #
    ######################

    # def train(self):
    #     pass

    ######################
    # modify star **     #
    ######################

    # def get_saver(self):
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def predict(self):
    #     pass

    def write_embedding_matrix(self,sess,data_path,file_name):
        # save embedding vec for words
        matrix = sess.run(self.embed_matrix)
        np.savetxt(data_path + file_name,matrix,fmt="%.4f",delimiter=',')




