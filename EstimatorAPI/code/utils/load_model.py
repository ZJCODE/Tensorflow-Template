# -*- coding: utf-8 -*-
"""
Created on 2018.10.29

@author: zhangjun
"""

import tensorflow as tf

class Model:
    """
    load tensorflow pd model
    """

    def __init__(self, sess, model_path):
        self.model_path = model_path
        self.sess = sess
        self.predictor = None
        self._load_model()

    def _load_model(self):
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], self.model_path)
        self.predictor = tf.contrib.predictor.from_saved_model(self.model_path)

    def predict(self, input_values_dict):
        output = self.predictor(input_values_dict)
        output_value = self._extract_predict_value(output)
        return output_value

    def _extract_predict_value(self,output):
        return output['prob']



if __name__ == '__main__':

    import glob
    model_path_list = glob.glob('../../model_files/export/*')
    model_path = max(model_path_list)
    # run | saved_model_cli show --dir export/1540797843 --all | to know the input
    input_values_dict_example = {'col1': [20],
                                 'col2': [33],
                                 'col3': [5],
                                 'col4': ["a"],
                                 'col5': [3],
                                 'col6': ["a#d"],
                                 'col7': ["f"],
                                 'col8': ["s"],
                                 }

    with tf.Session() as sess:
        model = Model(sess, model_path)
        out = model.predict(input_values_dict_example)
        print(out)
