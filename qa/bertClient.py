#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import os
import numpy as np
import csv

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data.csv'

'''
1默认问答系统方案:
输入问题与问答数据中待比较的所有问题,首先通过BERT表示为向量
然后用余弦相似度比较,得到语义最相似的问题与对应答案
'''


def getBestAnswer(qdata):
    b = np.load(path + "/data/question2vec1.npy")
    print('step1:导入问答数据中问题向量完成!!!,目前有效问题条数为:', len(b))
    bc = BertClient()
    testvec = bc.encode(["".join(qdata.split())])
    print('step2: 计算输入问题与问答数据中各个问题的相似度...')
    maxsimil = 0
    for i in range(1, len(b) + 1):
        simil_test_ques = cosine_similarity(b[i - 1], testvec[0])
        print('##############',simil_test_ques)
        if simil_test_ques > maxsimil:
            maxsimil = simil_test_ques
            index = i
            if maxsimil >= 0.999:
                break
    print('step3 获取最相似问题的相似度/索引:---->', maxsimil, index)

    similaryQuestion, bestAns = getSimilaryQuestionByIndex(index)
    print('问题是：', qdata)
    print('相似问题是：', similaryQuestion)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns


def cosine_similarity(vector1, vector2):
    '''计算余弦相似度'''
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return round(dot_product / ((normA ** 0.5) * (normB ** 0.5)), 2)


def getSimilaryQuestionByIndex(index):
    print('step4：通过索引从qa-clean-data中查询对应的相似问题及答案...', index)
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        id = 0
        for i in read:
            id += 1
            if id == index:
                simQuestion = i[0]
                ans = i[1]
    return simQuestion, ans


if __name__ == '__main__':
    print('开始查询相似问题--->')
    # testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    testQ = '上课注意力不集中怎么办？'
    # testQ = '如何提高学生上课注意力'
    # testQ = '学生沉迷游戏怎么办？'
    # testQ = '学生爱睡觉怎么办?'
    # testQ = '如何提高学习小组的讨论热情'
    getBestAnswer(testQ)
