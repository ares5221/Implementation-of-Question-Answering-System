#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
from bert_serving.client import BertClient
import numpy as np


# 读取atec数据集中q q1 q2将其encode后保存在对于的npy文件中

def papreData():
    file_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp1.csv'
    # 导入18668条数据
    bc = BertClient()
    setDataNum = 2000
    with open(file_dir, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        X = [[[] for i in range(2)] for j in range(setDataNum + 1)]
        index = 0
        for i in read:
            # print(i[0], i[1], i[2])
            tmp0 = bc.encode([i[0]])
            tmp1 = bc.encode([i[1]])
            tmp2 = bc.encode([i[2]])
            # print(tmp0, tmp1, tmp2)
            qq1_vec = np.append(tmp0, tmp1)
            qq2_vec = np.append(tmp0, tmp2)
            # print(qq1_vec == qq2_vec)
            X[index][0] = qq1_vec.tolist()
            X[index][1] = qq2_vec.tolist()
            index += 1
            if index % 100 == 0:
                print(index)
            if index > setDataNum:
                break
        X1 = np.array(X)
        np.save("x1_2000.npy", X1)
    print('数据导入10000条及预处理完成------------------')


if __name__ == '__main__':
    papreData()
