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

class model_template(model_base.model):

    def __init__(self,sess,batch_size,learning_rate,
                 opt_algo,epochs,regular_rate,dropout_rate,skip_step):

        super().__init__(sess,batch_size,learning_rate,
                         opt_algo,epochs,regular_rate,dropout_rate,skip_step)

        self.checkpoint_dir = 'checkpoint/template/model'

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

    # def inference(self):
    #     """
    #     all model structure is write in this function | which is very important
    #     usually we will get self.logit in this function
    #     """
    #     pass

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



