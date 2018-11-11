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

class deep_wide_model(model_base.model):

    def __init__(self,sess,batch_size,learning_rate,
                 opt_algo,epochs,regular_rate,dropout_rate,skip_step,
                 layer_size_list):

        super().__init__(sess,batch_size,learning_rate,
                         opt_algo,epochs,regular_rate,dropout_rate,skip_step)

        self.layer_size_list = layer_size_list # for deep part MLP 
        self.checkpoint_dir = 'checkpoint/deep_wide/model'


    ######################
    # modify star *      #
    ######################

    # def get_data(self):
    #     pass

    ######################
    # modify star **     #
    ######################

    # def get_label(self):
    #     pass

    ######################
    # modify star *****  #
    ######################

    def inference(self):

        with tf.variable_scope('embedding_layer'):

            if len(self.data_info.embedding_col) > 0:  # check for if there is some feature for embedding

                all_embedding_matrix  = {} # used for store all embeding matrix , we can visit each matrix by using feature name
                all_embedding_vec = {} # use for store all embedding vec

                for col in self.data_info.embedding_col:
                    # get embedding_matrix and embedding_vec for each kind of embedding feature 
                    embedding_matrix,embedding_vec = model_blocks.embedding_layer(id_feature_name = self.data_generator[col],
                                                                                  cate_size = self.data_info.category_summary_dict[col] + 1 , # do not forget to add one | index 0 for oov 
                                                                                  embedding_size = self.data_info.embedding_col_dict[col],
                                                                                  feature_name = col)
                    all_embedding_matrix[col] = embedding_matrix
                    all_embedding_vec[col] = embedding_vec

                embedding_input = [] # used for combine all embedding feature

                for vec in all_embedding_vec.values():
                    embedding_input.append(vec)

                embedding_input_tensor = tf.concat(embedding_input,axis=1)

        with tf.variable_scope('numerical_data'): 

            if len(self.data_info.numerical_col) > 0: # check if there is some numerical feature   

                numerical_input = [] # used for combine all numerical feature

                for col in self.data_info.numerical_col:
                    numerical_input.append(tf.reshape(self.data_generator[col],[-1,1]))
                numerical_input_tensor = tf.concat(numerical_input,axis=1)

        with tf.variable_scope('sparse_data'):

            got_sparse_input = True # used for denote whether we can get feature for model wide part 

            sparse_input = [] # used for combine all sparse feature / usually it's onehot feature

            if len(self.data_info.onehot_col) > 0:
                for col in self.data_info.onehot_col:
                    sparse_input.append(self.data_generator[col])

                sparse_input_tensor = tf.cast(tf.concat(sparse_input,axis=1),tf.float32)
            else:
                got_sparse_input = False 

        with tf.variable_scope('model_deep_part'):

            got_deep_input = True # used for denote whether we can get feature for model deep part 

            # if we got both embedding feature and numerical feature
            if len(self.data_info.embedding_col) > 0 and len(self.data_info.numerical_col) > 0:
                deep_input = tf.concat([embedding_input_tensor,numerical_input_tensor],axis=1)
            # if we only got embedding feature
            elif len(self.data_info.embedding_col) > 0:
                deep_input = embedding_input_tensor
            # if we only got numerical feature
            elif len(self.data_info.numerical_col) > 0:
                deep_input = numerical_input_tensor
            # if we got nothing for model deep part 
            else:
               got_deep_input = False

            if got_deep_input :
                # send deep part to a MLP 
                deep_output_tensor = model_blocks.multilayer_perceptron_layer(input_data = deep_input,
                                                               layer_size_list = self.layer_size_list,
                                                               regular_rate = self.regular_rate ,
                                                               dropout = self.dropout_rate ,
                                                               training = self.training)
 
        with tf.variable_scope('combine_deep_wide'):
            # combine model's deep and wide part 
            if got_deep_input and got_sparse_input:
                deep_wide_concat = tf.concat([deep_output_tensor,sparse_input_tensor],axis=1)
            elif got_deep_input:
                deep_wide_concat = deep_output_tensor
            elif got_sparse_input:
                deep_wide_concat = sparse_input_tensor
            else:
                raise ValueError(' Please Input Some Data ! You Know Nothing ! ')

            self.logits = tf.layers.dense(deep_wide_concat, 1 ,activation = None)


    ######################
    # modify star ***    #
    ######################

    # def model_loss(self):
    #     self.loss = model_blocks.get_loss(self.labels,self.logits)

    ######################
    # modify star ***    #
    ######################

    # def model_optimizer(self):
    #     pass

    ######################
    # modify star ****   #
    ######################

    def eval(self):
        self.prob = tf.sigmoid(self.logits)
        # add auc metrics
        self.auc,self.auc_update_op = tf.metrics.auc(self.labels,self.prob)

    ######################
    # modify star *      #
    ######################

    # def bulid(self):
    #   pass


    ######################
    # modify star ****   #
    ######################

    def train_one_epoch(self,epoch):
        start_time = time.time()
        self.sess.run(self.train_init_op)
        self.sess.run(tf.local_variables_initializer()) # local_variables_initializer is useful in particular for streaming metrics
        self.traininng = True
        total_loss = 0
        n_batches = 0
        step = 0 
        try:
            while True:
                _, l ,auc_val= self.sess.run([self.optimizer, self.loss,self.auc_update_op])
                if self.skip_step > 0: # if skip_step equal to 0 then do not show anything          
                    if (step + 1) % self.skip_step == 0:
                        print('Loss at step {0}: {1:.4f} | auc : {2:.4f}'.format(step, l,auc_val))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        auc = self.sess.run(self.auc)
        print('===========================================================')
        print('[Train] epoch : {0} | loss : {1:.4f} | auc : {2:.4f} | Took: {3:.2f} seconds'.format(epoch, total_loss/n_batches,auc,time.time() - start_time))


    ######################
    # modify star ****   #
    ######################

    def eval_once(self,epoch):
        start_time = time.time()
        self.sess.run(self.test_init_op)
        self.sess.run(tf.local_variables_initializer()) # local_variables_initializer is useful in particular for streaming metrics
        self.training = False
        total_loss = 0
        n_batches = 0
        try:
            while True:
                l,auc,_= self.sess.run([self.loss,self.auc,self.auc_update_op])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        auc = self.sess.run(self.auc)
        print('[Eval ] epoch : {0} | loss : {1:.4f} | auc : {2:.4f} | Took: {3:.2f} seconds'.format(epoch, total_loss/n_batches,auc,time.time() - start_time))
        print('===========================================================')


    ######################
    # modify star **     #
    ######################

    # def train(self):
    #     pass

    ######################
    # modify star **     #
    ######################

    def get_saver(self):
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

    ######################
    # modify star ****   #
    ######################

    # def predict(self):
    #     pass


