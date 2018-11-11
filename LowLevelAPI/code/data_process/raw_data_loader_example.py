# -*- coding: utf-8 -*-
"""
Created on 2018.8.13

@author: zhangjun
"""


import tensorflow as tf

data_path = 'raw_data_example'

dataset = tf.data.TextLineDataset(data_path)

def parse_row(raw_row):
	row_split = tf.sparse_tensor_to_dense(tf.string_split([raw_row],' '),default_value = '')
	feature = tf.sparse_tensor_to_dense(tf.string_split([row_split[0][0]],','),default_value = '')[0]
	print(feature)
	label= tf.string_to_number(row_split[0][1])
	row = {'feature':feature,'label':label}
	return row

dataset = dataset.map(parse_row)

dataset = dataset.padded_batch(4, padded_shapes={'feature':[None],'label':[]},padding_values = {'feature':'nan','label':tf.constant(0,tf.float32)})

i = dataset.make_one_shot_iterator()
data_gen = i.get_next()

with tf.Session() as sess:
	print(sess.run(data_gen))
	print(sess.run(data_gen))



