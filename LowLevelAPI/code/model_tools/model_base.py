# -*- coding: utf-8 -*-
"""
Created on 2018.8.8

@author: zhangjun
"""

import sys
sys.path.append("..")

import tensorflow as tf
import os
import utils.utils as utils  
import time
import model_tools.model_blocks as model_blocks
import data_process.data_loader as data_loader

class model:

    def __init__(self,sess,batch_size,learning_rate,
                 opt_algo,epochs,regular_rate,dropout_rate,skip_step):

        self.sess = sess
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.opt_algo = opt_algo
        self.epochs = epochs
        self.regular_rate = regular_rate
        self.dropout_rate = dropout_rate
        self.skip_step = skip_step
        self.checkpoint_dir = ''
        self.saver = None
        self.train_init_op = None
        self.test_init_op = None
        self.pred_init_op = None
        self.data_generator = None
        self.optimizer = None
        self.training = True
        self.data_info = None # contain numerical_col,onehot_col,embedding_col,embedding_col_dict,...
        self.has_test = False

    def get_data(self,data_path,train_data_name,test_data_name,pred_data_name,sep,low_freq_threshold):
        """
        get training data [dataset generator]
        """
        # train data 
        train_data = data_loader.Data(data_path = data_path,
                          data_name = train_data_name,
                          batch_size = self.batch_size,
                          is_train_period = True,
                          sep = sep,
                          low_freq_threshold = low_freq_threshold)
        train_data.get_dataset()
        train_dataset = train_data.dataset

        # data info
        self.data_info = train_data

        # iterator and init_op and data_generator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
        self.data_generator = iterator.get_next()
        self.train_init_op = iterator.make_initializer(train_dataset)

        # test data 
        if len(test_data_name) > 0:
            self.has_test = True
            test_data = data_loader.Data(data_path = data_path,
                              data_name = test_data_name,
                              batch_size = self.batch_size,
                              is_train_period = False,
                              sep = sep,
                              low_freq_threshold = low_freq_threshold)
            test_data.get_dataset()
            test_dataset = test_data.dataset
            self.test_init_op = iterator.make_initializer(test_dataset)

        # pred data 
        if len(pred_data_name) > 0:
            pred_data = data_loader.Data(data_path = data_path,
                              data_name = pred_data_name,
                              batch_size = self.batch_size,
                              is_train_period = False,
                              sep = sep,
                              low_freq_threshold = low_freq_threshold)
            pred_data.get_dataset()
            pred_dataset = pred_data.dataset
            self.pred_init_op = iterator.make_initializer(pred_dataset)


    def get_label(self):
        """
        set self.label in here
        """
        self.labels = tf.reshape(self.data_generator['labels'],[-1,1])

    def inference(self):
        """
        all model structure is write in this function | which is very important
        usually we will get self.logit in this function
        """
        numerical_input = []
        for col in self.data_info.numerical_col:
            numerical_input.append(tf.reshape(self.data_generator[col],[-1,1]))
        numerical_input_tensor = tf.concat(numerical_input,axis=1)
        

        self.logits = model_blocks.multilayer_perceptron_layer(input_data = numerical_input_tensor,
                                                               layer_size_list = [16,8,4,1],
                                                               regular_rate = self.regular_rate ,
                                                               dropout = self.dropout_rate ,
                                                               training = self.training)

    def model_loss(self):
        """
        write your own loss function for optimize 
        """
        self.loss = model_blocks.get_loss(self.labels,self.logits,type = 'sigmoid_cross_entropy')

    def model_optimizer(self):
        """
        get model opyimizer for model training
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = model_blocks.get_optimizer(self.opt_algo, self.learning_rate, self.loss)

    def eval(self): 
        """
        add more mertics for model evaluate
        """
        pass

    def bulid(self):
        """
        build model | get all kinds of elemnets for model tranining
        """
        self.get_label()
        self.inference()
        self.model_loss()
        self.model_optimizer()
        self.eval()

    def train_one_epoch(self,epoch):
        """
        tarin model by using training data , and print loss in each step  
        """
        start_time = time.time()
        self.sess.run(self.train_init_op)
        self.sess.run(tf.local_variables_initializer()) # local_variables_initializer is useful in particular for streaming metrics
        self.traininng = True
        total_loss = 0
        n_batches = 0
        step = 0 
        try:
            while True:
                _, l = self.sess.run([self.optimizer, self.loss])
                if self.skip_step > 0: # if skip_step equal to 0 then do not show anything                
                    if (step + 1) % self.skip_step == 0:
                        print('Loss at step {0}: {1:.4f}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('====================================================================')
        print('[Train] epoch : {0} | loss : {1:.4f} | Took: {2:.2f} seconds'.format(epoch, total_loss/n_batches,time.time() - start_time))

    def eval_once(self,epoch):
        """
        evaluate model on test data
        """
        start_time = time.time()
        self.sess.run(self.test_init_op)
        self.sess.run(tf.local_variables_initializer()) # local_variables_initializer is useful in particular for streaming metrics
        self.training = False
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l = self.sess.run(self.loss)
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        print('[Eval ] epoch : {0} | loss : {1:.4f} | Took: {2:.2f} seconds'.format(epoch, total_loss/n_batches,time.time() - start_time))
        print('====================================================================')

    def train(self,save=False,restore=False):
        """
        train model for several epochs and eval on test data if there exist one
        """
        self.sess.run(tf.global_variables_initializer())
        self.get_saver()

        if restore:
            self.restore_model()

        for epoch in range(1,self.epochs+1):
            self.train_one_epoch(epoch)
            if self.has_test:
                self.eval_once(epoch)
            if save:
                self.save_model(epoch)

    def get_saver(self):
        """
        get saver for save model 
        """
        self.saver = tf.train.Saver()

    def save_model(self,epoch):
        """
        save model to checkpoint_dir
        """
        utils.safe_mkdir(self.checkpoint_dir)
        self.saver.save(self.sess,self.checkpoint_dir,epoch)

    def restore_model(self):
        """
        restore model if there exist some saved model 
        """
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print("\nrestore model : {} \n".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("\nno model for restore\n")

    def predict(self,data_path):
        """
        return probability for binary classification problem 
        """
        self.sess.run(self.pred_init_op)
        self.training = False
        self.prob_result = tf.reshape(tf.sigmoid(self.logits),[-1])
        f = open(data_path + 'prob_result' ,'w')
        try:
            while True:
                prob_result = self.sess.run(self.prob_result)
                for v in prob_result:
                    f.write('{0:.8f}\n'.format(v))
        except tf.errors.OutOfRangeError:
            pass
        f.close()


