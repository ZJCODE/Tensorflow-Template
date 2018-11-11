# -*- coding: utf-8 -*-
"""
Created on 2018.8.11

@author: zhangjun
"""

# - windows_size [one side size]
# - way : skip_gram[center -> others] / cbow [others -> center]/ more 

sequence_data_path = '../sequence_data'
train_data_path = '../train_data'

train_data = open(train_data_path,'w',encoding = 'utf8')

windows_size = 2
way = 'skip_gram'

with open(sequence_data_path,'r',encoding = 'utf8') as f:
	for line in f.readlines():
		line_split = line.strip().split(' ')
		for i,w in enumerate(line_split):
			for distance in range(1,windows_size):
				if i - distance >=0:
					if way == 'skip_gram':
						train_data.write("{0} {1}\n".format(line_split[i],line_split[i - distance]))
					elif way == 'cbow':
						train_data.write("{0} {1}\n".format(line_split[i - distance],line_split[i]))
			for distance in range(1,windows_size):
				if i + distance < len(line_split):
					if way == 'skip_gram':
						train_data.write("{0} {1}\n".format(line_split[i],line_split[i + distance]))
					elif way == 'cbow':
						train_data.write("{0} {1}\n".format(line_split[i + distance],line_split[i]))
