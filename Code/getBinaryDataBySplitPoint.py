from random import shuffle
import numpy as np
from copy import deepcopy


def GetBinaryBySplitPoint(data, split_point_dict):
    result = {}
    for key in data.keys():
        if key == 'label':
            result['label'] = data[key]
            continue

        if key not in split_point_dict.keys() or len(split_point_dict[key]) == 0:
            df = data[key]
            length = len(df) * len(df.columns)
            df_array = df.values.reshape(length)
            df_array = df_array[np.isfinite(df_array)]
            shuffle(df_array)
            array = set(df_array[:1000])
            if array == {0 ,1}:
                key_type = 0
            else:
                key_type = -1
        elif isinstance(split_point_dict[key][0], list):
            key_type = 1
        elif isinstance(split_point_dict[key][0], float) or isinstance(split_point_dict[key][0], int):
            key_type = 2
        print(key, key_type)

        if key_type == -1:
            print("We don't have split points for key %s" % key)
            continue
        if key_type == 0:
            result[key] = data[key]
        elif key_type == 1:
            split_point = split_point_dict[key]
            for idx in range(len(split_point)):
                bool_df = data[key].isin(split_point[idx])
                df_tem = deepcopy(data[key])
                df_tem[bool_df] = 1
                df_tem[~bool_df] = 0
                result['%s%03d' % (key, idx)] = df_tem
        elif key_type == 2:
            split_point = split_point_dict[key]
            split_point.insert(0, -np.inf)
            split_point.append(np.inf)
            for idx in range(len(split_point ) -1):
                point_left = split_point[idx]
                point_right = split_point[idx +1]
                bool_df = (data[key] >= point_left) & (data[key] < point_right)
                df_tem = deepcopy(data[key])
                df_tem[bool_df] = 1
                df_tem[~bool_df] = 0
                result['%s%03d' % (key, idx)] = df_tem
        else:
            print("Cannot recognize key type %s" % (key_type))
    return result