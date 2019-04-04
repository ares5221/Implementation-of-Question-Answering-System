#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import numpy as np
from bert_serving.client import BertClient
import csv
import random
import xlrd
import os

'''
1,通过清理后的qa-clean-data.csv 构建question answer wronganswer数据集
构建的数据集保存在/data/AllquestAnsWroA.csv,通过该数据训练question2questionModel
2,通过清理后的数据集中的问题来构建question2vec将问题信息通过bert转化为向量,便于比较
3,通过清理后的数据集中的问题来构建answer2vec将问题信息通过bert转化为向量,便于比较
'''

path = os.path.abspath('../..')
filePath = path + '/data/qa-clean-data.csv'


def get_AllquestAnsWroA():
    # 收集所有答案便于选取任意答案作为错误答案
    allAns = []
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            # print(i[0], i[1])
            allAns.append(i[1])
    # 保存为csv
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            ranIndex = random.randint(0, len(allAns) - 1)
            wroAns = allAns[ranIndex]
            allQAA = [i[0], i[1], wroAns]
            with open(path + "/data/AllquestAnsWroA.csv", "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(allQAA)
        print('------->全部的问题/答案/错误答案数据已经保存完毕')


def getQAquestion():
    print('开始从qa-clean-data.csv中读取question data...')
    q_data = []  # 存放excel中获取的question 问题标题
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            q_data.append(i[0])
    print('获取到', len(q_data), '条问题')
    return q_data


def bertconvert(datas):
    question_list = []
    bc = BertClient()
    for i in range(0, len(datas)):
        data = datas[i]
        newdata = "".join(data.split())  # data.replace(' ', '')
        question_list.append(newdata)
    ss = bc.encode(question_list)
    np.save(path + "/data/question2vec1.npy", ss)


def getQAanswer():
    print('开始从qa-clean-data.csv中读取answer data...')
    a_data = []  # 存放excel中获取的question 问题标题
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            a_data.append(i[1])
    print('获取到', len(a_data), '条答案')
    return a_data


def bertconvert2(datas):
    answer_list = []
    bc = BertClient()
    for i in range(0, len(datas)):
        data = datas[i]
        newdata = "".join(data.split())
        answer_list.append(newdata)
    ss = bc.encode(answer_list)
    np.save(path + "/data/answer2vec.npy", ss)


if __name__ == '__main__':
    print('Task1 构建数据集AllquestAnsWroA.csv')
    if 1 == 0:
        get_AllquestAnsWroA()
    print('Task1 Finish OK--->')

    print('Task2 构建数据集question2vec1.npy')
    if 1 == 0:
        data = getQAquestion()
        bertconvert(data)
    print('Task2 Finish OK--->')

    print('Task3 构建数据集answer2vec.npy')
    if 1 == 0:
        data = getQAanswer()
        bertconvert2(data)
    print('Task3 Finish OK--->')
