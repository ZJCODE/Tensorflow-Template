# -*- coding: utf-8 -*-
"""
Created on 2018.8.9

@author: zhangjun
"""

import os
import tensorflow as tf
import argparse
import utils.utils as utils
from model_collections.deep_wide_model import deep_wide_model
from model_collections.word2vec_model import word2vec_model
from model_collections.sequence_model import sequence_model


parser = argparse.ArgumentParser()

parser.add_argument('-gpu_num',         default='0', type=str, help='witch gpu to use')
parser.add_argument('-batch_size',      default=2048, type=int, help='batch size')
parser.add_argument('-learning_rate',   default=0.001, type=float, help='learning rate')
parser.add_argument('-opt_algo',        default='rmsprop', type=str, help='optimize algorithm')
parser.add_argument('-epochs',          default=10, type=int, help='epochs num')
parser.add_argument('-regular_rate',    default=0, type=float, help='regular rate')
parser.add_argument('-dropout_rate',    default=0, type=float, help='dropout rate')
parser.add_argument('-skip_step',       default=100, type=float, help='skip step')
parser.add_argument('-layer_size_list', default='64,32,16', type=str, help='layer size list for MLP')

parser.add_argument('-train_data_name', default='train_data', type=str, help='train data name')
parser.add_argument('-test_data_name', default='test_data', type=str, help='test data name')
parser.add_argument('-pred_data_name', default='pred_data', type=str, help='pred data name')
parser.add_argument('-data_path', default='../data/', type=str, help='data path')

parser.add_argument('-save', default='False', type=str, help='save or not')
parser.add_argument('-restore', default='False', type=str, help='restore or not')

parser.add_argument('-num_sampled', default=3, type=int, help='negative num_ ampled')
parser.add_argument('-vocab_embed_size', default=128, type=int, help='vocab embedding size')

parser.add_argument('-low_freq_threshold', default=1, type=int, help='cate low freq threshold')

parser.add_argument('-model', default='deep_wide', type=str, help='restore or not')

parser.add_argument('-csv_sep', default=' ', type=str, help='csv sep')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_num


def main():

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    model = sequence_model(sess = sess,
                                    batch_size = 2,
                                    learning_rate = args.learning_rate,
                                    opt_algo = args.opt_algo,   
                                    epochs = args.epochs,
                                    regular_rate = args.regular_rate,
                                    dropout_rate = args.dropout_rate,
                                    skip_step = args.skip_step,
                                    hidden_size_list = [64,32,16],
                                    sequence_embedding_size = 32)


    model.get_data(data_path = args.data_path,
               train_data_name = args.train_data_name,
               test_data_name = '',
               pred_data_name = '',
               sep = args.csv_sep,
               low_freq_threshold = args.low_freq_threshold)

    model.bulid()


    model.train(save = utils.str2bool(args.save),
             restore = utils.str2bool(args.restore))


if __name__ == '__main__':
    main()



