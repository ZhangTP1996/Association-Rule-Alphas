import pandas as pd
import json
import numpy as np
from copy import deepcopy
import os
from load_data import load_daily_data_simple
import datetime

def get_conf(result_data, key_rule, label):
    sup = 0
    ret = 0
    for rule in key_rule:
        feature_list = []
        for item in rule:
            feature_list.append(result_data[item])
        df = (feature_list[0] == 1)
        for feature in feature_list[1:]:
            df = df & (feature == 1)
        label_tem = label.copy()
        df = df.reindex(label_tem.index)
        label_tem[~(df == True)] = np.nan
        sup += df.sum().sum()
        ret += label_tem.sum().sum()
    if sup < 0.1:
        return 0
    return float(ret) / sup

def get_interval_conf(result_data, data, key, key_rule, label, left_point, right_point):
    data_tem = deepcopy(data)
    bool_df = (data_tem >= left_point) & (data_tem < right_point)
    data_tem[bool_df] = 1
    data_tem[~bool_df] = 0
    result_data[key] = data_tem
    conf = get_conf(result_data, key_rule, label)
    return conf, data_tem

def compare_conf(conf_left, conf_right, conf_baseline):
    if conf_left > conf_baseline:
        if conf_right > conf_baseline:
            if conf_left > conf_right:
                return 1
            else:
                return 2
        else:
            return 1
    elif conf_right > conf_baseline:
        return 2
    return 0

def adjust_key(args, result_data, key, key_rule, label, accuracy=0.002, epochs=10):
    conf_baseline = get_conf(result_data, key_rule, label)
    with open(args.splitPointPath, 'r') as f:
        split_point_dict = json.load(fp=f)
    key_name = key[:-3]
    key_num = int(key[-3:])
    data = load_daily_data_simple(args.dataPath, ['%s'%key_name])
    data = data[key_name]
    array = data.values.reshape(len(data) * len(data.columns))
    array = array[np.isfinite(array)]
    array = list(sorted(array))
    accuracy_length = int(len(array) * accuracy)
    if key_num == 0:
        left_point = -np.inf
        right_point = split_point_dict[key_name][key_num]
        if len(key_rule) == 0:
            return [[-np.inf, right_point],
                     [-np.inf, right_point]]
        for epoch in range(epochs):
            try:
                right_idx = array.index(right_point)
            except:
                right_idx = int(len(array)*(key_num+1)*0.1)
                if int(array[right_idx]) == int(right_point):
                    pass
                else:
                    return [[-np.inf, right_point],
                            [-np.inf, right_point]]

            # 右→右
            right_point = array[right_idx + accuracy_length]
            conf_right, data_tem_right_1 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                             right_point)

            if conf_right > conf_baseline:
                result_data[key] = data_tem_right_1
                right_point = array[right_idx + accuracy_length]
                continue


            # 右→左
            right_point = array[right_idx - accuracy_length]
            conf_right, data_tem_right_2 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                             right_point)

            if conf_right > conf_baseline:
                result_data[key] = data_tem_right_2
                right_point = array[right_idx - accuracy_length]
            else:
                right_point = array[right_idx]
                break
        return [[-np.inf, split_point_dict[key_name][key_num]],
                 [-np.inf, right_point]]
    elif key_num == len(split_point_dict[key_name]):
        left_point = split_point_dict[key_name][key_num - 1]
        right_point = np.inf
        if len(key_rule) == 0:
            return [[left_point, np.inf],
                     [left_point, np.inf]]
        for epoch in range(epochs):
            try:
                left_idx = array.index(left_point)
            except:
                left_idx = int(len(array)*(key_num)*0.1)
                if int(array[left_idx]) == int(left_point):
                    pass
                else:
                    return [[left_point, np.inf],
                             [left_point, np.inf]]

            # 左→左
            left_point = array[left_idx - accuracy_length]
            conf_left, data_tem_left_1 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                           right_point)


            if conf_left > conf_baseline:
                result_data[key] = data_tem_left_1
                left_point = array[left_idx - accuracy_length]
                continue

            # 左→右
            left_point = array[left_idx + accuracy_length]
            conf_left, data_tem_left_2 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                           right_point)


            if conf_left > conf_baseline:
                result_data[key] = data_tem_left_2
                left_point = array[left_idx + accuracy_length]
            else:
                left_point = array[left_idx]
                break
        return [[split_point_dict[key_name][key_num - 1], np.inf],
                 [left_point, np.inf]]
    else:
        left_point = split_point_dict[key_name][key_num-1]
        right_point = split_point_dict[key_name][key_num]
        if len(key_rule) == 0:
            return [[left_point, right_point],
                    [left_point, right_point]]
        for epoch in range(epochs):
            try:
                left_idx = array.index(left_point)
                right_idx = array.index(right_point)
            except:
                left_idx = int(len(array)*(key_num)*0.1)
                right_idx = int(len(array)*(key_num+1)*0.1)
                if int(array[left_idx]) == int(left_point) and int(array[right_idx]) == int(right_point):
                    pass
                else:
                    return [[left_point, right_point],
                             [left_point, right_point]]

            # 左→左
            left_point = array[left_idx - accuracy_length]
            right_point = array[right_idx]
            conf_left, data_tem_left_1 = get_interval_conf(result_data, data, key, key_rule, label, left_point, right_point)

            # 右→右
            left_point = array[left_idx]
            right_point = array[right_idx + accuracy_length]
            conf_right, data_tem_right_1 = get_interval_conf(result_data, data, key, key_rule, label, left_point, right_point)

            flag = compare_conf(conf_left, conf_right, conf_baseline)
            if flag == 1:
                result_data[key] = data_tem_left_1
                left_point = array[left_idx - accuracy_length]
                right_point = array[right_idx]
                continue
            elif flag == 2:
                result_data[key] = data_tem_right_1
                left_point = array[left_idx]
                right_point = array[right_idx + accuracy_length]
                continue


            # 左→右
            left_point = array[left_idx + accuracy_length]
            right_point = array[right_idx]
            conf_left, data_tem_left_2 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                           right_point)

            # 右→左
            left_point = array[left_idx]
            right_point = array[right_idx - accuracy_length]
            conf_right, data_tem_right_2 = get_interval_conf(result_data, data, key, key_rule, label, left_point,
                                                             right_point)

            flag = compare_conf(conf_left, conf_right, conf_baseline)
            if flag == 0:
                left_point = array[left_idx]
                right_point = array[right_idx]
                break
            elif flag == 1:
                result_data[key] = data_tem_left_2
                left_point = array[left_idx + accuracy_length]
                right_point = array[right_idx]
            elif flag == 2:
                result_data[key] = data_tem_right_2
                left_point = array[left_idx]
                right_point = array[right_idx - accuracy_length]
        return [[split_point_dict[key_name][key_num-1], split_point_dict[key_name][key_num]],
                 [left_point, right_point]]




def adjust_interval(args, binary_data, label, ruleList, elogger):
    split_point_dict_new = {}
    # result_data = binary_data.copy()
    with open(args.splitPointPath, 'r') as f:
        split_point_dict = json.load(fp=f)
    key_list = list(binary_data.keys())
    i = 1
    start = datetime.datetime.now()
    for count, key in enumerate(key_list):
        if count == int(len(key_list) * 0.1 * i):
            end = datetime.datetime.now()
            elogger.log("%d0 percent has been adjusted with %d seconds" % (i, (end-start).seconds))
            start = datetime.datetime.now()
            i += 1
        if key[-3:].isdigit():
            key_name = key[:-3]
        else:
            continue
        if isinstance(split_point_dict[key_name][0], float):
            pass
        else:
            continue
        key_rule = []
        for rule in ruleList:
            if key in rule:
                key_rule.append(rule)
        # split_point_record = adjust_key(args, result_data, key, key_rule, label)
        split_point_record = adjust_key(args, binary_data, key, key_rule, label)
        split_point_dict_new[key] = split_point_record
    with open(os.path.join('./save/split_point_adjusted.json'), 'w') as f:
        json.dump(split_point_dict_new, f)
