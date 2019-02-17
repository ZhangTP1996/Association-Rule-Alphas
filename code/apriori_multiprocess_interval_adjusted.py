from itertools import combinations
import numpy as np
import argparse
from logger import Logger
import os
import json
from getBinaryDataBySplitPoint import GetBinaryBySplitPoint
from load_data import load_daily_data_simple
from read_txt_new import get_attr
import multiprocessing
import traceback
import datetime
from random import shuffle
from adjust_interval import adjust_interval

num_process = 12

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type = str)
parser.add_argument('--labelPath', type = str)
parser.add_argument('--splitPointPath', type = str)
parser.add_argument('--sup', type = float)
parser.add_argument('--conf', type = float)
parser.add_argument('--saveFile', type = str)
parser.add_argument('--logger', type = str)

args = parser.parse_args()
def process_candidateList(candidateList, ruleList):
    returnList = []
    if len(candidateList) != 0:
        length = len(candidateList[0])
    else:
        return []
    for rule in candidateList:
        _subsets = map(list, [x for x in combinations(rule, length-1)])
        flag = 1
        for element in _subsets:
            if element in ruleList:
                continue
            else:
                flag = 0
        if flag == 1:
            returnList.append(rule)
    return returnList

def multiprocess_candidateList(candidateList, ruleList):
    pool = multiprocessing.Pool(processes=num_process)
    returnList = []
    results = []
    tempList = []
    for i in range(num_process):
        tempList.append([])
    shuffle(candidateList)
    length_tem = int(len(candidateList) / num_process)
    for count in range(num_process):
        if count == num_process - 1:
            tempList[count] = candidateList[count * length_tem:]
        else:
            tempList[count] = candidateList[count * length_tem: (count + 1) * length_tem]
    for i in range(num_process):
        print(len(tempList[i]))
        res = pool.apply_async(process_candidateList, (tempList[i], ruleList, ))
        results.append(res)
    pool.close()
    pool.join()
    for res in results:
        returnList.extend(res.get())
    return returnList

def get_candidate_List(ruleList, elogger):
    if len(ruleList) == 0:
        return []
    ruleList = sorted(ruleList)
    candidateList = []
    length = len(ruleList[0])
    start = datetime.datetime.now()
    if length == 1:
        for idx in range(len(ruleList)):
            for rule in ruleList[idx+1:]:
                rule_new = ruleList[idx].copy()
                rule_new.append(rule[0])
                candidateList.append(rule_new)
        elogger.log("The number of candidateList with length %d is %d" % (length+1, len(candidateList)))
        return candidateList
    else:
        for idx in range(len(ruleList)):
            for rule in ruleList[idx+1:]:
                rule_new = ruleList[idx].copy()
                if rule_new[-2] == rule[-2]:
                    rule_new.append(rule[-1])
                    candidateList.append(rule_new)
                else:
                    break
    end = datetime.datetime.now()
    elogger.log("Time spent in finding candidateList: %d seconds" % (end-start).seconds)
    elogger.log("The number of candidateList with length %d is %d" % (length+1, len(candidateList)))

    start = datetime.datetime.now()
    if len(candidateList) == 0:
        return []
    returnList = multiprocess_candidateList(candidateList, ruleList)
    end = datetime.datetime.now()
    elogger.log("Time spent in filtering candidateList: %d seconds" % (end-start).seconds)
    return returnList

def returnRuleWithMinSupport(binary_data_path, label, ruleList, minSupport, minConfidence, maxLength, file_path, length):
    binary_data = load_daily_data_simple(binary_data_path)
    if length == 5:
        minSupport_return = minSupport + 0.005
    else:
        minSupport_return = minSupport
    try:
        f = open(file_path, 'a+')
        returnList = []
        for rule in ruleList:
            df = (binary_data[rule[0]] == 1)
            for idx in range(1, len(rule)):
                df = df & (binary_data[rule[idx]] == 1)
            sup = np.sum(np.sum(df))
            if (sup / maxLength) >= minSupport_return:
                returnList.append(rule)
                label_tem = label.copy()
                label_tem[~(df == True)] = np.nan
                ret = np.sum(np.sum(label_tem))
                if ret / sup >= minConfidence and (sup / maxLength) >= minSupport:
                    f.write('%s with sup %.6lf and conf %.6lf\n' % (str(rule), sup / maxLength, ret / sup))
                    f.flush()
        f.close()
    except Exception as e:
        print(traceback.format_exc())
    del binary_data
    return returnList


def runApriori(binary_data, binary_data_path, label, minSupport, minConfidence, elogger):
    maxLength = np.sum(np.sum(np.isfinite(label)))
    itemList = binary_data.keys()
    itemList = [[item] for item in itemList]

    candidateList = get_candidate_List(itemList, elogger)
    elogger.log("The number of candidateList filtered is %d" % len(candidateList))

    length = 2
    while(len(candidateList) >= 1):

        start = datetime.datetime.now()
        pool = multiprocessing.Pool(processes=num_process)
        ruleList = []
        results = []
        tempList = []
        for i in range(num_process):
            tempList.append([])

        length_tem = int(len(candidateList) / num_process)
        for count in range(num_process):
            if count == num_process - 1:
                tempList[count] = candidateList[count * length_tem :]
            else:
                tempList[count] = candidateList[count * length_tem : (count+1) * length_tem]
        for i in range(num_process):
            print(len(tempList[i]))
            label_tem = label.copy()
            file_path = args.saveFile
            res = pool.apply_async(returnRuleWithMinSupport, (binary_data_path, label_tem, tempList[i], minSupport, minConfidence, maxLength, file_path, length, ))
            results.append(res)
        pool.close()
        pool.join()
        for res in results:
            ruleList.extend(res.get())
        end = datetime.datetime.now()
        elogger.log("The number of rules with length %d is %d" % (length, len(ruleList)))

        # 调整因子的区间
        if length == 2:
            print("Start adjusting interval.")
            adjust_interval(args, binary_data, label, ruleList, elogger)
            for key in binary_data.keys():
                binary_data[key].to_csv('./binary_data_adjusted/%s.csv' % str(key))
            binary_data_path = './binary_data_adjusted'
        length += 1


        elogger.log('Time spent in filtering ruleList: %d seconds' % (end-start).seconds)
        candidateList = get_candidate_List(ruleList, elogger)
        elogger.log("The number of candidateList filtered is %d" % len(candidateList))

if __name__ == '__main__':
    data = load_daily_data_simple(args.dataPath)
    label = load_daily_data_simple(args.labelPath, ['label'])
    label = label['label']
    label = label.loc[label.index <= '20171231']
    minSup = args.sup
    minConf = args.conf
    elogger = Logger(args.logger)
    elogger.log(str(os.getpid()))
    elogger.log(str(args._get_kwargs()))

    with open('./EquiDepth_Label000_pct5.json', 'r') as f:
        split_point_dict = json.load(fp=f)
    binary_data = GetBinaryBySplitPoint(data, split_point_dict)

    attr_set = get_attr()

    attr_set = list(attr_set)

    binary_data_path = './binary_data'
    for key in binary_data.keys():
        binary_data[key].to_csv('./binary_data/%s.csv' % key)
    runApriori(binary_data, binary_data_path, label, minSup, minConf, elogger)
