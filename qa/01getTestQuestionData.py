#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import csv
from bertClientRun2 import getBestAnswer2bySimilyQuestionByq2qModel
from bertClient import getBestAnswer

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data.csv'
save_log_dir = path + '/log/Fun2Log.csv'
save_log_dir1 = path + '/log/Fun1Log.csv'


def getTestQ():
    qdata = []
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            qdata.append(i[0])
    print(len(qdata), '-->', qdata)
    qset = set(qdata)
    print(qset)
    qqlist = list(qset)
    print(len(qqlist), '-->', qqlist)
    return qqlist[0:100]


if __name__ == '__main__':
    print('开始查询相似问题--->')
    data = getTestQ()
    # for i in range(100):
    #     ss = data[i]
    #     res1, res2 = getBestAnswer2bySimilyQuestionByq2qModel(ss)
    #     news = [ss, res1]
    #     with open(save_log_dir, 'a', newline='', encoding='utf-8') as csvfile:
    #         spamwriter = csv.writer(csvfile)
    #         spamwriter.writerow(news)
    for i in range(100):
        ss = data[i]
        res1, res2 = getBestAnswer(ss)
        news = [ss, res1]
        with open(save_log_dir1, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
    print('测试问题与相似问题已经保存在log文件中!!!!!')
