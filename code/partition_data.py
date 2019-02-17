import pandas as pd
import numpy as np
from config import config
from copy import deepcopy
import json
from random import shuffle
from load_data import load_daily_data_simple

def get_type(df):
    length = len(df) * len(df.columns)
    df_array = df.values.reshape(length)
    df_array = df_array[np.isfinite(df_array)]
    shuffle(df_array)
    array = set(df_array[:1000])
    if array == {0,1}:
        return 0
    elif len(array) < 30:
        array = set(df_array)
        if len(array) < 30:
            return 1
        else:
            return 2
    else:
        return 2



def Equi_Depth_split(feature_array, depth=config['depth']):
    i = 1
    split_point = []
    while(i*depth < 1):
        split_point.append(feature_array[int(len(feature_array) * i * depth)])
        i += 1
    return split_point


def categorical_to_binary(df, key):
    # input: 一个dataframe
    # output: 几个binary的dataframe组成的dict
    length = len(df) * len(df.columns)
    category_list = list(sorted(set(df.values.reshape(length)[np.isfinite(df.values.reshape(length))])))
    result = {}
    count = 0
    sup_tem = 0
    split_point = []
    split_point_record = []
    for category in category_list:
        print(category)
        support_now = np.sum(np.sum(df == category)) / np.sum(np.sum(np.isfinite(df)))
        if sup_tem == 0:
            bool_df = (df == category)
        else:
            bool_df = bool_df | (df == category)
        sup_tem += support_now
        split_point_record.append(category)
        if sup_tem < config['minsup_categorical']:
            continue
        else:
            sup_tem = 0
            df_tem = deepcopy(df)
            df_tem[bool_df] = 1
            df_tem[~bool_df] = 0
            result['%s%03d'%(key, count)] = df_tem
            print(split_point_record)
            split_point.append(split_point_record)
            split_point_record = []
            count += 1
    if len(split_point_record) != 0:
        df_tem = deepcopy(df)
        df_tem[bool_df] = 1
        df_tem[~bool_df] = 0
        result['%s%03d' % (key, count)] = df_tem
        print(split_point_record)
        split_point.append(split_point_record)
        count += 1
    return result, split_point

def quantitative_to_binary(df, label, key):
    # input: 一个dataframe
    # output: 几个binary的dataframe组成的dict
    result = {}
    feature_array = np.array(df.values.reshape(len(df) * len(df.columns)))
    label_array = np.array(label.values.reshape(len(label) * len(label.columns)))
    bool_array = np.isfinite(feature_array)
    feature_array = feature_array[bool_array]
    label_array = label_array[bool_array]
    if len(feature_array) != len(label_array):
        print("The length of feature and label is not the same!")
        exit()
    index = np.lexsort((label_array, feature_array))
    feature_array = feature_array[index]

    feature_array = feature_array[np.isfinite(feature_array)]
    feature_array = sorted(feature_array)
    split_point = Equi_Depth_split(feature_array)

    point_left = -np.inf
    split_point.append(np.inf)
    count = 0
    split_point_record = []
    for point_right in split_point:
        df_tem = deepcopy(df)
        bool_df = (df_tem >= point_left) & (df_tem < point_right)
        df_tem[bool_df] = 1
        df_tem[~bool_df] = 0
        support = np.sum(np.sum(df_tem)) / np.sum(np.sum(np.isfinite(df_tem)))
        if support < config['minsup_quantitative']:
            print(point_left, point_right, support)
            continue
        split_point_record.append(point_right)
        point_left = point_right
        result['%s%03d' % (key, count)] = df_tem
        print("Support for feature %s%03d is %lf" % (key, count, support))
        count += 1
    return result, split_point_record[:-1]


def get_binary(data):
    # 返回两个dict，其中每一个key是一个binary的csv文件
    # 另一个保存所有分割点，比如{'feature1':[[1],[2],[3,4]], 'feature2':[0.5,1.2,3.3]}
    result = {}
    split_point_list = {}
    for key in data.keys():
        print("Processing feature %s." % key)
        key_type = get_type(data[key])
        print("Processing feature %s. It's type %d" % (key, key_type))
        if key_type == 0:
            result[key] = data[key]
        elif key_type == 1:
            result_tem, split_point_tem = categorical_to_binary(data[key], key)
            result.update(result_tem)
            split_point_list[key] = split_point_tem
        elif key_type == 2:
            result_tem, split_point_tem = quantitative_to_binary(data[key], data['label'], key)
            result.update(result_tem)
            split_point_list[key] = split_point_tem
    with open(config['split_point_save_path'], 'w') as f:
        json.dump(split_point_list, f)
    return result

if __name__ == '__main__':
    path = '/home/user/AssociationRule_1/2018-12-02/TrainDataZZ15'
    data = load_daily_data_simple(path)
    label = load_daily_data_simple('/home/user/AssociationRule_1/2018-12-02/data', ['label'])
    label['label'] = label['label'].loc[label['label'].index <= '20171231']
    for key in data.keys():
        data[key] = data[key].loc[label['label'].index]
    data['label'] = label['label']
    result = get_binary(data)
    print(len(result.keys()))
