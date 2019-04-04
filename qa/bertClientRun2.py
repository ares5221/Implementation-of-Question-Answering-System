#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import xlrd, os
import numpy as np
import tensorflow as tf
from calSimilarityByq2qModel import calSimilarityByq2qModel
from getAnswerByq2aModel import getAnswerByq2aModel

path = os.path.abspath('..')
filePath = path + '/data/qa-all-data.xlsx'

#  2通过一个训练好的问题到问题的多层感知机来比较找到与输入问题最相似的问题
def getBestAnswer2bySimilyQuestionByq2qModel(qdata):
    datapath = os.path.abspath('..')
    b = np.load(datapath + "/data/question2vec1.npy")
    print('step 3:load question vector form question2vec1 success!!!!!!!!!', len(b))
    bc = BertClient()
    testvec = bc.encode(["".join(qdata.split())])
    print('step4: cal cosine_similarity by q2q mlp model')
    tf.reset_default_graph()
    index, maxsimil = calSimilarityByq2qModel(b, testvec)
    print('step4 获取到最相似问题的索引@@@@@@@@', maxsimil, index)
    index = index + 2  # 由于表中数据索引从第二列开始
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    if sheet.cell(index, 2).value is not '':
        similyQue = sheet.cell(index, 2).value
    else:
        similyQue = sheet.cell(index, 1).value
    print('在问答数据中查找得到相似问题是', similyQue)

    bestAns = getAnsByIndex(index)
    print('问题是：', qdata)
    print('回答是：', bestAns)
    return bestAns



def getAnsByIndex(index):
    print('step5：通过问题索引从excel文档中获取对应的答案...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    if sheet.cell(index, 13).value is not '':
        ans = sheet.cell(index, 13).value
    else:
        ans = '该问题正在讨论中...'
    return ans

if __name__ == '__main__':
    # print('START------111111111111111111')
    # testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    # testQ = '上课注意力不集中怎么办？'
    testQ = '如何提高学生表达能力'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    getBestAnswer2bySimilyQuestionByq2qModel(testQ)