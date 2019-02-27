# -*- coding: utf-8 -*-
"""
Created on 2018.12.05

@author: zhangjun
"""

import numpy as np
import multiprocessing
import tensorflow as tf
from tensorflow.contrib.lookup import HashTable
from tensorflow.contrib.lookup import KeyValueTensorInitializer
import glob


class dataProcess:

    def __init__(self, params):
        self.sequence_cols = params['sequence_cols']
        self.map_extend_features = params['map_extend_features']
        self.hash_table = {}

    def squence_split(self, raw_features):
        if len(self.sequence_cols) > 0:
            for col, sep in self.sequence_cols:
                raw_features = self.several_values_columns_to_array(raw_features, col, sep)
        return raw_features

    def map_more_features(self, raw_features):
        if len(self.map_extend_features) > 0:
            for key, map_file, map_cols, sep in self.map_extend_features:
                self.build_hash_table(map_file, map_cols[1:], sep)  # [1:] skip first col which is map_id
                raw_features = self.extend_more_feature(raw_features, key, map_cols[1:])
        return raw_features

    def do_process(self, raw_features):
        raw_features = self.squence_split(raw_features)
        raw_features = self.map_more_features(raw_features)
        return raw_features

    @staticmethod
    def several_values_columns_to_array(raw_features, feature_name, sep):
        raw_features[feature_name] = tf.sparse_tensor_to_dense(
            tf.string_split(raw_features[feature_name], sep),
            default_value='')
        return raw_features

    def build_hash_table(self, map_file, map_cols, sep):
        # first col of map_file must be map_id
        keys = []  # map_id col
        values = dict(zip(map_cols, [[] for _ in range(len(map_cols))]))

        with open(map_file, 'r', encoding='utf8') as f:
            for line in f:
                line_split = line.rstrip().split(sep)
                keys.append(line_split[0].encode('utf8'))
                for i, col in enumerate(map_cols):
                    values[col].append(line_split[i + 1].encode('utf8'))

        for col in map_cols:
            default_val = '-'
            table = HashTable(
                KeyValueTensorInitializer(keys, values[col]), default_val)
            self.hash_table[col] = table

    def extend_more_feature(self, raw_features, key, map_cols):
        for col in map_cols:
            raw_features[key + '_' + col] = self.hash_table[col].lookup(raw_features[key])
        return raw_features


class dataStat:
    """

    notice : must run by python3

    params example :
        params = {'cols':['col1','col2','col3'],'dtypes':['num','string','num'],'sep':',',select_cols:['col1','col2']}

    usage :
        fs = FeatureStat(file_path, pool_num, params,select_cols)

        print(fs.string_stat)
        print(fs.num_stat)

        if you want to calculate statistics on one specific file such as 'data/file_1' then you can run
        string_stat,num_stat,num_stat_temp = fs.doStat('data/file_1')

    result example : when select_cols = cols
        string_stat
            {'col2': {'a': 8, 'e': 4, 'd': 4, 'c': 4}}
        num_stat
            {'col1': {'mean': 3.8, 'std': 2.052476, 'max': 6.0, 'min': 1.0},
            'col3': {'mean': 3.6, 'std': 0.805047, 'max': 5.0, 'min': 3.0}}

    result example : when select_cols = ['col1','col2']
        string_stat
            {'col2': {'a': 8, 'e': 4, 'd': 4, 'c': 4}}
        num_stat
             {'col1': {'mean': 3.8, 'std': 2.052476, 'max': 6.0, 'min': 1.0}}
    """

    def __init__(self, file_path, pool_num, params, is_path=True):
        self.file_path = file_path
        if is_path:
            self.file_list = glob.glob(self.file_path + '/*')
        else:
            self.file_list = [file_path]
        self.pool_num = pool_num
        self.cols = params['cols']
        self.dtypes = params['dtypes']
        self.sep = params['sep']
        self.select_cols = set(params['select_cols'])
        self.sequence_cols = params['sequence_cols']  # params['sequence_cols'] -> [['col6', '#'], ['col8', '#']]
        self._map_reduce()

    def doStat(self, file):
        num_stat = {}
        string_stat = {}
        num_stat_temp = {}
        for col, dtype in zip(self.cols, self.dtypes):
            if col not in self.select_cols:
                continue
            if dtype == 'string':
                string_stat[col] = {}
            elif dtype == 'num':
                num_stat[col] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
                num_stat_temp[col] = {'sum': 0, 'sum_square': 0, 'cnt': 0, 'max': -1e10, 'min': 1e10}
            else:
                print('wrong dtype')

        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                line_split = line.rstrip().split(self.sep)
                for index, (col, dtype) in enumerate(zip(self.cols, self.dtypes)):
                    if col not in self.select_cols:
                        continue
                    if dtype == 'string':
                        if col in self.sequence_cols:
                            values = line_split[index].split(self.sequence_cols[col])
                        else:
                            values = [line_split[index]]
                        for value in values:
                            value = value.strip()
                            if value not in string_stat[col]:
                                string_stat[col][value] = 1
                            else:
                                string_stat[col][value] += 1
                    elif dtype == 'num':
                        value = float(line_split[index])
                        num_stat_temp[col]['sum'] += value
                        num_stat_temp[col]['sum_square'] += value * value
                        num_stat_temp[col]['cnt'] += 1
                        num_stat_temp[col]['max'] = max(value, num_stat_temp[col]['max'])
                        num_stat_temp[col]['min'] = min(value, num_stat_temp[col]['min'])
                    else:
                        pass

            for col, dtype in zip(self.cols, self.dtypes):
                if col not in self.select_cols:
                    continue
                if dtype == 'num':
                    num_stat[col]['max'] = num_stat_temp[col]['max']
                    num_stat[col]['min'] = num_stat_temp[col]['min']
                    value_sum = num_stat_temp[col]['sum']
                    value_cnt = num_stat_temp[col]['cnt']
                    value_sum_square = num_stat_temp[col]['sum_square']
                    num_stat[col]['mean'] = value_sum / value_cnt
                    num_stat[col]['std'] = round(
                        np.sqrt((value_sum_square - value_sum * value_sum / value_cnt) / (value_cnt - 1)), 6)

        return string_stat, num_stat, num_stat_temp

    def _map(self):
        pool = multiprocessing.Pool(self.pool_num)
        results = pool.map(self.doStat, self.file_list)
        pool.close()
        pool.join()
        return results

    def _reduce(self, results):
        num_stat = {}
        string_stat = {}
        num_stat_temp = {}
        for col, dtype in zip(self.cols, self.dtypes):
            if col not in self.select_cols:
                continue
            if dtype == 'num':
                num_stat[col] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

        for index, (string_stat_part, _, num_stat_temp_part) in enumerate(results):
            if index == 0:
                string_stat = string_stat_part
                num_stat_temp = num_stat_temp_part
            else:
                for col, dtype in zip(self.cols, self.dtypes):
                    if col not in self.select_cols:
                        continue
                    if dtype == 'string':
                        for k, v in string_stat_part[col].items():
                            if k not in string_stat[col]:
                                string_stat[col].update({k: v})
                            else:
                                string_stat[col][k] += v
                    elif dtype == 'num':
                        num_stat_temp[col]['sum'] += num_stat_temp_part[col]['sum']
                        num_stat_temp[col]['sum_square'] += num_stat_temp_part[col]['sum_square']
                        num_stat_temp[col]['cnt'] += num_stat_temp_part[col]['cnt']
                        num_stat_temp[col]['max'] = max(num_stat_temp[col]['max'], num_stat_temp_part[col]['max'])
                        num_stat_temp[col]['min'] = max(num_stat_temp[col]['min'], num_stat_temp_part[col]['min'])

                    else:
                        pass
        for col, dtype in zip(self.cols, self.dtypes):
            if col not in self.select_cols:
                continue
            if dtype == 'num':
                num_stat[col]['max'] = num_stat_temp[col]['max']
                num_stat[col]['min'] = num_stat_temp[col]['min']
                value_sum = num_stat_temp[col]['sum']
                value_cnt = num_stat_temp[col]['cnt']
                value_sum_square = num_stat_temp[col]['sum_square']
                num_stat[col]['mean'] = round(value_sum / value_cnt, 6)
                num_stat[col]['std'] = round(
                    np.sqrt((value_sum_square - value_sum * value_sum / value_cnt) / (value_cnt - 1)), 6)

        return string_stat, num_stat

    def _map_reduce(self):
        results = self._map()
        self.string_stat, self.num_stat = self._reduce(results)


if __name__ == '__main__':
    pass
