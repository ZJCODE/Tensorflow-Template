# -*- coding: utf-8 -*-
"""
Created on 2018.11.2

@author: zhangjun
"""

import glob, multiprocessing
import numpy as np

class FeatureStat:
    """

    notice : must run by python3
    
    params example :
        params = {'cols':['col1','col2','col3'],'dtypes':['num','string','num'],'sep':','}

    usage :
        fs = FeatureStat(file_path, pool_num, params,select_cols)

        print(fs.string_stat)
        print(fs.num_stat)

        if you want to calculate statistics on one specific file such as 'data/file_1' then you can run
        string_stat,num_stat,num_stat_temp = fs.doStat('data/file_1')

    result example : when select_cols = params['cols']
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

    def __init__(self, file_path, pool_num, params, select_cols, sequence_cols={}):
        self.file_path = file_path
        self.file_list = glob.glob(self.file_path + '/*')
        self.pool_num = pool_num
        self.params = params
        self.select_cols = set(select_cols)
        self.sequence_cols = sequence_cols
        self._map_reduce()

    def doStat(self, file):
        num_stat = {}
        string_stat = {}
        num_stat_temp = {}
        for col, dtype in zip(self.params['cols'], self.params['dtypes']):
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
            for line in f.readlines():
                line_split = line.strip().split(self.params['sep'])
                for index, (col, dtype) in enumerate(zip(self.params['cols'], self.params['dtypes'])):
                    if col not in self.select_cols:
                        continue
                    if dtype == 'string':
                        if col in self.sequence_cols:
                            values = line_split[index].split(self.sequence_cols[col])
                        else:
                            values = [line_split[index]]
                        for value in values:
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

            for col, dtype in zip(self.params['cols'], self.params['dtypes']):
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
        for col, dtype in zip(self.params['cols'], self.params['dtypes']):
            if col not in self.select_cols:
                continue
            if dtype == 'num':
                num_stat[col] = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

        for index, (string_stat_part, _, num_stat_temp_part) in enumerate(results):
            if index == 0:
                string_stat = string_stat_part
                num_stat_temp = num_stat_temp_part
            else:
                for col, dtype in zip(self.params['cols'], self.params['dtypes']):
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
        for col, dtype in zip(self.params['cols'], self.params['dtypes']):
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
    import config

    file_path = config.DATA_PATH + '/train'
    print(file_path)
    params = {'cols': ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'],
              'dtypes': ['num', 'num', 'num', 'string', 'num', 'string', 'string', 'string'],
              'sep': ','}
    select_cols = params['cols']
    sequence_cols = {'col6': '#', 'col8': '#'}

    fs = FeatureStat(file_path, pool_num=4, params=params, select_cols=select_cols, sequence_cols=sequence_cols)

    print(fs.string_stat)
    print(fs.num_stat)

    # print(fs.doStat(file_path + '/train_data'))
