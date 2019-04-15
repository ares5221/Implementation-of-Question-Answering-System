#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import os
import numpy as np
import tensorflow as tf
from bertClient import cosine_similarity
from bertClient import getSimilaryQuestionByIndex
from getAnswerByq2aModel import getAnswerByq2aModel

path = os.path.abspath('..')
filePath = path + '/data/qa-all-data.xlsx'


#  3通过余弦相似度来比较找到与输入问题最相似的5个问题
#   然后通过训练好的问题到答案的MLP通过上一步得到的最相似问题来找到对应的答案
def getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(qdata):
    print('step1导入问题答案数据--->')
    q2v = np.load(path + "/data/question2vec1.npy")
    a2v = np.load(path + "/data/answer2vec.npy")
    bc = BertClient()
    queryvec = bc.encode(["".join(qdata.split())])
    print('step2获取到5个最相似问题的索引--->')
    simlist, idlist = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    for i in range(1, len(q2v)+1):
        res = cosine_similarity(q2v[i - 1], queryvec[0])
        if res > min(simlist):
            simlist[simlist.index(min(simlist))] = res
            idlist[simlist.index(min(simlist))] = i
    print('问题相似度:', simlist)
    print('相似问题ID:', idlist)
    similaryQuestion, bestAns = getSimilaryQuestionByIndex(idlist[0])
    print('step3在问答数据中查找得到相似问题是---->', similaryQuestion)

    ## 从5个候选答案中选取最好的答案
    ansMaxSimi = 0
    ansId = 0
    for j in range(len(idlist)):
        ansvec = a2v[idlist[j]]
        tf.reset_default_graph()
        anssimilary = getAnswerByq2aModel(a2v, ansvec)
        print('#########', anssimilary)
        if anssimilary > ansMaxSimi:
            ansMaxSimi = anssimilary
            ansId = idlist[j]
    xx, bestAns = getSimilaryQuestionByIndex(ansId)
    print('#######在问答数据中查找得到最佳答案的问题是---->', xx)
    print('问题是：', qdata)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns



if __name__ == '__main__':
    testQ = '如何提高学生表达能力'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(testQ)
