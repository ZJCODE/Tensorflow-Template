# -*- coding: utf-8 -*-
"""
Created on 2018.8.13

@author: zhangjun
"""

import sys
sys.path.append("..")

import numpy as np
import utils.similar as similar


class vecs:
    def __init__(self):

        self.vecs = []
        self.meta = []

    def get_vecs(self):
        return self.vecs
    def get_meta(self):
        return self.meta

    def get_meta_dict(self):
        self.meta_dict = ''

    def find_most_similar(self,vec_name,top_num = 5,sim_type = 'cos'):
        distance = False
        try:
            vec = self.vecs[self.meta_dict[vec_name],:]
        except:
            print('{} is not in dict'.format(vec_name))
        if sim_type == 'cos':
            sim = similar.cos_similarity(vec,self.vecs)
        elif sim_type == 'manhattan':
            sim = similar.manhattan_distance_similarity(vec,self.vecs,distance)
        elif sim_type == 'euclidean':
            sim = similar.euclidean_distance_similarity(vec,self.vecs,distance)
        else:
            print('use cos similarity as default')
            sim = similar.cos_similarity(vec,self.vecs)
        if distance:
            sim_order = np.argsort(sim)[:top_num]
        else:
            sim_order = np.argsort(-1*sim)[:top_num]
            
        return ([self.meta_dict_reverse[x] for x in sim_order],sim[sim_order])


class vecs_embedding(vecs):
    """docstring for vecs_embedding"""
    def __init__(self):
        super().__init__()

    def load_vecs(self,vec_path,sep = ' '):
        with open(vec_path,'r',encoding = 'utf8')  as f:
            for line in f.readlines():
                line_split = line.strip().split(sep)
                self.vecs.append(np.array(line_split).astype(float))
        self.vecs = np.array(self.vecs)

    def load_meta(self,meta_path,sep = ' '):
        self.reindex = []
        with open(meta_path,'r',encoding = 'utf8') as f:
            for line in f.readlines():
                line_split = line.strip().split(sep)
                self.meta.append(line_split[0])
                self.reindex.append(line_split[1])
        self.meta = np.array(self.meta)
        self.reindex = np.array(self.reindex).astype(int)

    def get_meta_dict(self):
        self.meta_dict = dict(zip(self.meta,self.reindex))
        self.meta_dict_reverse = dict(zip(self.reindex,self.meta))
        self.meta_dict_reverse.update({0:'<default>'})


class vecs_from_line_model(vecs):
    """docstring for vecs_line"""
    def __init__(self):
        super().__init__()

    def load_meta_vecs(self,meta_vec_path , sep = ' '):
        with open(meta_vec_path,'r',encoding = 'utf8')  as f:
            for line in f.readlines():
                line_split = line.strip().split(sep)
                self.meta.append(line_split[0])
                self.vecs.append(np.array(line_split[1:]).astype(float))
        self.meta = np.array(self.meta)
        self.vecs = np.array(self.vecs)
        
    def get_meta_dict(self):
        self.meta_dict = dict(zip(self.meta,range(len(self.meta))))
        self.meta_dict_reverse = dict(zip(range(len(self.meta)),self.meta))


def main():

    # #-------line start-------
    # vec_line = vecs_from_line_model()
    # vec_line.load_meta_vecs('line_data_example')
    # vec_line.get_meta_dict()
    # #-------line  end -------

    vec_line = vecs_embedding()
    vec_line.load_vecs('../../data_word2vec/vocab_embedding_matrix',',')
    vec_line.load_meta('../../data_word2vec/vocabulary_words_reindex',',')
    vec_line.get_meta_dict()


    # print(vec_line.get_meta())
    # print(vec_line.get_vecs())
    # print(vec_line.meta_dict)
    # print(vec_line.meta_dict_reverse)

    name,sim = vec_line.find_most_similar('s',5,'cos')
    for n,s in zip(name,sim):
        print('{0} :{1:.3f}'.format(n,s))


if __name__ == '__main__':
    main()

