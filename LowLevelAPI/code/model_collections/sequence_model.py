# -*- coding: utf-8 -*-
"""
Created on 2018.8.9

@author: zhangjun
"""

import sys
sys.path.append("..")
import time
import tensorflow as tf 
import model_tools.model_base as model_base
import model_tools.model_blocks as model_blocks

class sequence_model(model_base.model):

    def __init__(self,sess,batch_size,learning_rate,
                 opt_algo,epochs,regular_rate,dropout_rate,skip_step,
                 hidden_size_list,sequence_embedding_size):

        super().__init__(sess,batch_size,learning_rate,
                         opt_algo,epochs,regular_rate,dropout_rate,skip_step)

        self.hidden_size_list = hidden_size_list
        self.sequence_embedding_size = sequence_embedding_size
        self.checkpoint_dir = 'checkpoint/sequence_model/model'

    ######################
    # modify star *      #
    ######################

    # def get_data(self):
    #     """
    #     get training data [dataset generator]
    #     """
    #     pass

    ######################
    # modify star **     #
    ######################

    # def get_label(self):
    #     """
    #     set self.label in here
    #     """
    #     pass

    ######################
    # modify star *****  #
    ######################

    def inference(self):
        """
        all model structure is write in this function | which is very important
        usually we will get self.logit in this function
        """
        with tf.variable_scope('lstm_layer'):
            lstm_outputs_list = []
            for col in self.data_info.sequence_col:
                seq_inputs = self.data_generator[col]
                print('seq_inputs',seq_inputs)
                embedding_matrix,seq_inputs_vec = model_blocks.embedding_layer(id_feature_name = seq_inputs,
                                                                                  cate_size = self.data_info.sequence_summary_dict[col] + 1 , # do not forget to add one | index 0 for oov 
                                                                                  embedding_size = self.sequence_embedding_size,
                                                                                  feature_name = col)

                print(seq_inputs_vec)
                seq_inputs_length = self.data_generator[col+'_length']

                print(seq_inputs_length)

                lstm_outputs,state = model_blocks.dynamic_lstm_layer(inputs = seq_inputs_vec,
                                                                sequence_length = seq_inputs_length,
                                                                hidden_size_list = self.hidden_size_list)
                print(lstm_outputs)

                lstm_outputs_last = tf.transpose(lstm_outputs,[0,2,1])[:,:,-1]
                
                print(lstm_outputs_last)

        #self.logits = tf.layers.dense(lstm_outputs_last, 1 ,activation = None)




    ######################
    # modify star ***    #
    ######################

    # def model_loss(self):
    #     """
    #     write your own loss function for optimize 
    #     """
    #     pass

    ######################
    # modify star ***    #
    ######################

    # def model_optimizer(self):
    #     """
    #     get model opyimizer for model training
    #     """
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def eval(self):
    #     """
    #     add more mertics for model evaluate
    #     """
    #     pass

    ######################
    # modify star *      #
    ######################

    # def bulid(self):
    #     """
    #     build model | get all kinds of elemnets for model tranining
    #     """
    #   pass

    ######################
    # modify star ****   #
    ######################

    # def train_one_epoch(self,epoch):
    #     """
    #     tarin model by using training data , and print loss in each step  
    #     """
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def eval_once(self,epoch):
    #     """
    #     evaluate model on test data
    #     """
    #     pass

    ######################
    # modify star **     #
    ######################

    # def train(self):
    #     """
    #     train model for several epochs and eval on test data if there exist one
    #     """
    #     pass

    ######################
    # modify star **     #
    ######################

    # def get_saver(self):
    #     """
    #     get saver for save model 
    #     """
    #     pass

    ######################
    # modify star ****   #
    ######################

    # def predict(self):
    #     """
    #     return predict result
    #     """
    #     pass



