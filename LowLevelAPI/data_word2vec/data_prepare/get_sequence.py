# -*- coding: utf-8 -*-
"""
Created on 2018.8.11

@author: zhangjun
"""

import glob
import re

doc_path = '../docs/'
sequence_file_path = '../sequence_data'
docs = glob.glob(doc_path + "*")

sequence_file = open(sequence_file_path,'w',encoding = 'utf8')

for doc in docs:
	with open(doc,'r',encoding = 'utf8') as f:
		for line in f.readlines():
				line_split = re.split('[,.;?|"]',line.strip())
				for l in line_split:
					if len(l.split(" ")) > 2:
						sequence_file.write(l.strip() + '\n')


