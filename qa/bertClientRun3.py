#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import os
import numpy as np
import tensorflow as tf
from bertClient import getSimilaryQuestionByIndex
from getAnswerByq2aModel import getAnswerByq2aModel

path = os.path.abspath('..')
# filePath = path + '/data/qa-all-data.xlsx'


#  3通过余弦相似度来比较找到与输入问题最相似的5个问题
#   然后通过训练好的问题到答案的MLP通过上一步得到的最相似问题来找到对应的答案
def getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(qdata):
    print('step1导入问题答案数据--->')
    doc_vecs = np.load(path + "/data/question2vec1.npy")
    a2v = np.load(path + "/data/answer2vec.npy")
    bc = BertClient()
    query_vec = bc.encode(["".join(qdata.split())])

    topk = 5
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    idlist = [x + 1 for x in topk_idx]  # qa-clean-data中问题索引是从1开始的
    print('step2获取到5个最相似问题的相似度及索引--->')
    for idx in idlist:
        print('> %s\t%s' % (score[idx], idx))
    similaryQuestion, bestAns = getSimilaryQuestionByIndex(idlist[0])
    print('step3在问答数据中查找得到相似问题是---->', similaryQuestion)

    # 从5个候选答案中选取最好的答案
    ansMaxSimi = 0
    ansId = 0
    for j in range(len(idlist)):
        ansvec = a2v[idlist[j]]
        tf.reset_default_graph()
        anssimilary = getAnswerByq2aModel(a2v, ansvec)
        if anssimilary > ansMaxSimi:
            ansMaxSimi = anssimilary
            ansId = idlist[j]
    xx, bestAns = getSimilaryQuestionByIndex(ansId)
    print('在问答数据中查找得到最佳答案的问题是---->', xx)
    print('问题是：', qdata)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns, xx



if __name__ == '__main__':
    testQ = '如何提高学生表达能力'
    # testQ = '如何阻止学生打架'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(testQ)
