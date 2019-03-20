#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

path = os.path.abspath('../..')


def get_atecQuestAns():
    file_dir = path + '/data/atec_nlp.csv'
    # 导入204746条数据
    saveatecdata = []
    with open(file_dir, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            if len(i) > 3:
                print(i[1], i[2], i[3])
                allqa = [i[1], i[2], i[3]]
                saveatecdata.append(allqa)
    return saveatecdata


def getData():
    datasize = 200000
    X = [[] for i in range(datasize)]
    Y = [[] for i in range(datasize)]
    data = get_atecQuestAns()
    bc = BertClient()
    for index in range(datasize):
        tmp = data[index]
        v1 = bc.encode([tmp[0]])
        v2 = bc.encode([tmp[1]])
        qq1_vec = np.append(v1, v2)
        X[index] = qq1_vec.tolist()
        Y[index] = int(tmp[2])
        if index % 100 == 0:
            print(index, 'is finish')
            print(tmp[0], tmp[1], tmp[2])
    X_train = np.array(X)
    Y_train = np.array(Y)
    np.save(path + '/data/Y_atec_200000.npy', Y_train)
    np.save(path + '/data/X_atec_200000.npy', X_train)
    print(X_train.shape)
    print(Y_train.shape)
    print('save x train')


# Start Position----------->>>>>>>>>
if __name__ == '__main__':
    print(path)
    getData()
