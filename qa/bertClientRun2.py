#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
from bert_serving.client import BertClient
import xlrd, os
import numpy as np
import tensorflow as tf
from calSimilarityByq2qModel import calSimilarityByq2qModel

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data.csv'

'''
2通过一个训练好的问题-问题的多层感知机来比较找到与输入问题最相似的问题
'''


def getBestAnswer2bySimilyQuestionByq2qModel(qdata):
    b = np.load(path + "/data/question2vec1.npy")
    print('step1:导入问答数据中问题向量完成!!!,目前有效问题条数为:', len(b))
    bc = BertClient()
    testvec = bc.encode(["".join(qdata.split())])
    print('step2:通过预训练的MLP计算相似度 by q2q mlp model')
    tf.reset_default_graph()

    index, maxsimil = calSimilarityByq2qModel(b, testvec)
    print('step3:获取到最相似问题的索引为:--->', index, maxsimil)
    fun2index = index[0]
    similaryQuestion, bestAns = getSimilaryQuestionByIndex(fun2index)
    print('问题是：', qdata)
    print('相似问题是:', similaryQuestion)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns


def getSimilaryQuestionByIndex(index):
    print('step4:通过索引从qa-clean-data中查询对应的相似问题及答案...', index)
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
    # testQ = '注意力不集中怎么办？'
    # testQ = '如何让孩子喜欢学英语'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    # testQ = '如何处理学生恋爱问题'
    # testQ = '如何防止孩子抄作业'
    testQ = '孩子学习没有热情怎么办'
    getBestAnswer2bySimilyQuestionByq2qModel(testQ)
