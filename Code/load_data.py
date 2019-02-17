import pandas as pd
import os


def load_daily_data_simple(path_data, var_list=None):
    data = {}
    if var_list is None:
        full_files = os.listdir(path_data)
        var_list = [item[:-4] for item in full_files]
    for var_name in var_list:
        var = pd.read_csv(os.path.join(path_data, '%s.csv' % var_name))
        var['date'] = [str(item) for item in list(var['date'])]
        var.set_index('date', inplace=True)
        var.index = pd.to_datetime(var.index).strftime("%Y%m%d")
        var.index.name = 'date'
        data[var_name] = var
    return data


