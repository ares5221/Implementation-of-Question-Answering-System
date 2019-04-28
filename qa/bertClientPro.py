#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import os
import numpy as np
import csv
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import timeit

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data425.csv'

'''
1默认问答系统方案:
输入问题与问答数据中待比较的所有问题,首先通过BERT表示为向量
然后用余弦相似度比较,得到语义最相似的问题与对应答案
优化代码，提高响应速度。响应时间0.35s左右，目前问答数据对400条左右
'''


def clock(func):
    def clocked(*args):
        t0 = timeit.default_timer()
        result = func(*args)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        print('该方法消耗时间情况如下：', '[%0.8fs] -> %s ' % (elapsed, name))
        return result

    return clocked


# @clock
def getBestAnswerByKDTree(qdata):
    doc_vecs = np.load(path + "/data/question2vec1.npy")
    print('step1:导入问答数据中问题向量完成!!!,目前有效问题条数为:', len(doc_vecs))
    bc = BertClient()
    query_vec = bc.encode(["".join(qdata.split())])[0]
    print('step2: 计算输入问题与问答数据中各个问题的相似度 by normalized dot product...')
    topk = 5
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    for idx in topk_idx:
        print('> %s\t%s' % (score[idx], idx))
    questionID = topk_idx[0] + 1  # qa-clean-data中问题索引是从1开始的
    similaryQuestion, bestAns = getSimilaryQuestionByIndex(questionID)
    print('问题是：', qdata)
    print('相似问题是：', similaryQuestion)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns


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


def testKD_Tree(doc_vecs, query_vec):
    '''
    通过KD-Tree及BallTree的方式来加快搜索速度。从向量结合中快速找到最'相似'的向量
    采用欧式距离，曼哈顿距离，闵科夫斯基距离等方式，但是找到的最相近向量与我们通过余弦相似度比较得到的向量并不一样
    doc_vecs:向量集合
    query_vec:待比较的向量
    :return:dist:相似度  index:查找的最相近向量的索引
    '''
    print('构建KD tree')
    k_dtree = KDTree(doc_vecs, leaf_size=2)
    dist, index = k_dtree.query(query_vec, k=3)
    print('#################', dist, index)
    print('构建Ball tree')
    ball_tree = BallTree(doc_vecs, leaf_size=2)
    dist, index = ball_tree.query(query_vec, k=3)
    print('#################', dist, index)


if __name__ == '__main__':
    print('开始查询相似问题--->')
    # testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    # testQ = '上课注意力不集中怎么办？'
    # testQ = '如何提高学生上课注意力'
    # testQ = '学生沉迷游戏怎么办？'
    # testQ = '学生爱睡觉怎么办?'
    testQ = '如何阻止学生打架'
    # testQ = '如何提高学习小组的讨论热情'
    # testQ = '学生说谎该怎么处理更合适'
    getBestAnswerByKDTree(testQ)
