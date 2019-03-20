#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv
import os
from bert_serving.client import BertClient

path = os.path.abspath('../..')


def get_atecQuestAns():
    file_dir = path + '/data/AllquestAnsWroA.csv'
    # 导入5858条数据
    saveatecdata = []
    with open(file_dir, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            allqa = [i[0], i[1], i[2]]
            saveatecdata.append(allqa)
        # print('over--------------------------------', len(saveatecdata))
    return saveatecdata


def getData():
    datasize = 10000
    X = [[] for i in range(datasize)]
    Y = [0 for i in range(datasize)]
    data = get_atecQuestAns()
    bc = BertClient()
    for index in range(0, 10000, 2):
        tmp = data[int(index / 2)]
        print(tmp[0], tmp[1], tmp[2])
        v1 = bc.encode([tmp[0]])
        v2 = bc.encode([tmp[1]])
        v3 = bc.encode([tmp[2]])
        qq1_vec = np.append(v1, v2)
        qq2_vec = np.append(v1, v3)
        print(qq2_vec)
        X[index] = qq1_vec.tolist()
        X[index + 1] = qq2_vec.tolist()
        Y[index] = 1
        Y[index + 1] = 0
        if index % 100 == 0:
            print(index, 'is finish')
    X_train = np.array(X)
    Y_train = np.array(Y)
    np.save(path + '/data/Y_qa_all_data.npy', Y_train)
    np.save(path + '/data/X_qa_all_data.npy', X_train)
    print(X_train.shape)
    print(Y_train.shape)
    print('save x train')


# Start Position----------->>>>>>>>>
if __name__ == '__main__':
    print(path)
    getData()
