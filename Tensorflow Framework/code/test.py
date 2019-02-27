# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

from utils.model import get_embedding_weight
from utils.conf import Config

config = Config(base_dir='../conf')

def test_get_weight(clean=True):
    get_embedding_weight(checkpoint_path=config.get_model_prop('model_check_dir'),
                         var_output_path='../data/weight',
                         clean=clean)

if __name__ == '__main__':
    test_get_weight(False)
