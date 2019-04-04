#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
import csv, os
from bert_serving.client import BertClient

path = os.path.abspath('../..')

'''
通过atec_nlp.csv数据构建用于trainQuestion2QuestionModel的数据
生成的数据保存在X_atec_50000.npy  Y_atec_50000.npy
模型迭代600次,经过比较X_atec_100000训练精度在0.85, X_atec_50000精度在0.877,所以优先用50000的数据训练
'''


def get_atecQuestAns():
    file_dir = path + '/data/atec_nlp.csv'
    # 导入18668条数据
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
    datasize = 100000
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
    np.save(path + '/data/Y_atec_100000.npy', Y_train)
    np.save(path + '/data/X_atec_100000.npy', X_train)
    print(X_train.shape)
    print(Y_train.shape)
    print('save atec_nlp train')


# Start Position----------->>>>>>>>>
if __name__ == '__main__':
    print(path)
    getData()
