# -*- coding: utf-8 -*-
"""
Created on 2018.9.29

@author: zhangjun
"""

import os
import argparse
import glob
import time
import tensorflow as tf
import shutil
import config
from model import model_fn, train_and_eval, train
from utils.export_model import export_model


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir', type=str, default=config.MODEL_CHECK_DIR, help='Base directory for the model.')
parser.add_argument(
    '--model_export_dir', type=str, default=config.MODEL_EXPORT_DIR, help='directory for export model.')
parser.add_argument(
    '--num_threads', type=str, default=config.NUM_PARALLEL, help='CPU threads used.')
parser.add_argument(
    '--train_epochs', type=int, default=2, help='Number of training epochs.')
parser.add_argument(
    '--learning_rate', type=float, default=0.01, help='Learning rate for model.')
parser.add_argument(
    '--opt_algo', type=str, default='Adam', help='optimization algorithm.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=1, help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=2048, help='Number of examples per batch.')
parser.add_argument(
    '--train_data', type=str, default=config.TRAIN_DATA_PATH, help='Path to the train data.')
parser.add_argument(
    '--eval_data', type=str, default=config.EVAL_DATA_PATH, help='Path to the validation data.')
parser.add_argument(
    '--keep_train', type=int, default=0, help='Whether to keep training on previous trained model [0/1] .')
parser.add_argument(
    '--log_steps', type=int, default=100, help='log step.')
parser.add_argument(
    '--layer_size_list', type=str, default='128,128,128', help='layer_size_list.')
parser.add_argument(
    '--regular_rate', type=float, default=0.0, help='regular rate.')
parser.add_argument(
    '--dropout', type=float, default=0.0, help='dropout.')
parser.add_argument(
    '--gpu_num', default='3', type=str, help='witch gpu to use')
parser.add_argument(
    '--train_type', default='train', type=str, help='with eval or not')
args = parser.parse_args()


def main():
    # Clean up the model directory if not keep training
    if not args.keep_train:
        shutil.rmtree(args.model_dir, ignore_errors=True)
        print('Remove model directory: {}'.format(args.model_dir))

    if not config.LOCAL_TRAIN:
        rsync_model_files = glob.glob(config.RSYNC_MODEL_DIR + "/*")
        print("export models we have :",rsync_model_files)

        for model_file in rsync_model_files:
            model_time = int(model_file.split('/')[-1])
            if model_time < int(time.time()) - 60*60*24 * 7:
                print("delete :" , model_file)
                shutil.rmtree(model_file, ignore_errors=True)

    # Clean up the model export directory
    shutil.rmtree(args.model_export_dir, ignore_errors=True)

    # Set Which GPU to use or do not use gpu
    if args.gpu_num == '-1':
        session_config = tf.ConfigProto(device_count={'CPU': args.num_threads})
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

        session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                        device_count={'CPU': args.num_threads})

    # Set Model Params
    model_params = {
        'learning_rate': args.learning_rate,
        'layer_size_list': list(map(lambda x: int(x), args.layer_size_list.split(","))),
        'regular_rate': args.regular_rate,
        'dropout': args.dropout,
        'opt_algo': args.opt_algo
    }

    estimator_config = tf.estimator.RunConfig().replace(session_config=session_config,
                                                        log_step_count_steps=args.log_steps,
                                                        save_summary_steps=args.log_steps)

    model = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=args.model_dir,
                                   params=model_params,
                                   config=estimator_config)

    print("\n=========================================")
    print("train type : ", args.train_type)
    for embedding in config.FILE_EMBEDDING_COLUMNS:
        print("{} embedding size : {}".format(embedding[0],embedding[-1]))
    print("train at GPU : ", args.gpu_num)
    print("learning rate : ", args.learning_rate)
    print("optimize algorithm : ", args.opt_algo)
    print("batch size : ", args.batch_size)
    print("epochs : ", args.train_epochs)
    print("layer size list : ", args.layer_size_list)
    print("regular rate : ", args.regular_rate)
    print("dropout rate : ", args.dropout)
    print("=========================================")
    print("model saved at : ", config.MODEL_EXPORT_DIR)
    print("=========================================\n")

    if args.train_type == 'train_and_eval':
        train_and_eval(model=model,
                       train_data=args.train_data,
                       eval_data=args.eval_data,
                       train_epochs=args.train_epochs,
                       batch_size=args.batch_size,
                       epochs_per_eval=args.epochs_per_eval)
    else:
        train(model=model,
              train_data=args.train_data,
              train_epochs=args.train_epochs,
              batch_size=args.batch_size)

        export_model(model, args.model_export_dir)


if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.logging.set_verbosity(0)
    main()
