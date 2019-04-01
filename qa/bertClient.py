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


def read_excel():
    print('开始从excel文档中read data...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    q_data = []  # 存放excel中获取的question 问题标题
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
        else:
            question = sheet.cell(i, 1).value
        # ansName = sheet.cell(i, 9)  # 存放excel中的 回答者name
        q_data.append(question)
    return q_data


def bertconvert(datas):
    question_list = []
    bc = BertClient()
    print(len(datas))
    for i in range(0, len(datas)):
        data = datas[i]
        newdata = "".join(data.split())  # data.replace(' ', '')
        question_list.append(newdata)
    ss = bc.encode(question_list)
    np.save(path + "/data/question2vec1.npy", ss)


#  1默认问题答案匹配方案,通过生成的向量间余弦相似度比较,得到最相似的问题,再得到对应答案
def getBestAnswer(qdata):
    b = np.load(path + "/data/question2vec1.npy")
    print('step 3:load question vector success!!!!!!!!!!!!!!!!', len(b))
    bc = BertClient()
    testvec = bc.encode(["".join(qdata.split())])
    print('step4: cal cosine_similarity')
    maxsimil = 0
    for i in range(len(b)):  # 3625
        simil_test_ques = cosine_similarity(b[i], testvec[0])
        if simil_test_ques > maxsimil:
            maxsimil = simil_test_ques
            index = i
    print('step4 获取到最相似问题的索引@@@@@@@@', maxsimil, index)
    index = index + 2  # 由于表中数据索引从第二列开始
    similaryQuestion = getSimilaryQuestionByIndex(index)
    bestAns = getAnsByIndex(index)
    print('问题是：', qdata)
    print('相似问题是：', similaryQuestion)
    print('回答是：', bestAns)
    return similaryQuestion, bestAns


def cosine_similarity(vector1, vector2):
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
    # print('step5：通过问题索引从excel文档中获取对应的答案...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    if sheet.cell(index, 2).value is not '':
        simQuestion = sheet.cell(index, 2).value
    else:
        simQuestion = sheet.cell(index, 1).value
    return simQuestion


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


#  2通过一个训练好的问题到问题的多层感知机来比较找到与输入问题最相似的问题
def getBestAnswer2bySimilyQuestionByq2qModel(qdata):
    datapath = os.path.abspath('..')
    b = np.load(datapath + "/data/question2vec1.npy")
    print('step 3:load question vector form qa-all-data.xlsx success!!!!!!!!!', len(b))
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


#  3通过一个训练好的问题到问题的多层感知机来比较找到与输入问题最相似的问题
#   然后通过训练好的问题到答案的MLP通过上一步得到的最相似问题来找到对应的答案
def getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(qdata):
    q2v = np.load(path + "/data/question2vec1.npy")
    a2v = np.load(path + "/data/answer2vector.npy")
    bc = BertClient()
    queryvec = bc.encode(["".join(qdata.split())])
    # 获取相似问题
    tf.reset_default_graph()
    index, maxsimil = calSimilarityByq2qModel(q2v, queryvec)
    print('step3 获取到最相似问题的索引--->>', maxsimil, index)
    index = index + 2  # 由于表中数据索引从第二列开始
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    if sheet.cell(index, 2).value is not '':
        similyQue = sheet.cell(index, 2).value
    else:
        similyQue = sheet.cell(index, 1).value
    print('step3 在问答数据中查找得到相似问题是---->', similyQue)
    # 获取对应答案
    ansvec = q2v[index - 2]
    tf.reset_default_graph()
    ansindex, anssimilary = getAnswerByq2aModel(a2v, ansvec)
    if sheet.cell(ansindex + 2, 13).value is not '':
        bestAns = sheet.cell(ansindex + 2, 13).value
    else:
        bestAns = '相关答案正在编辑中...'
    print('step4 在问答数据中查找得到答案是---->', bestAns)
    print('问题是：', qdata)
    print('回答是：', bestAns)
    return bestAns


if __name__ == '__main__':
    # print('START------111111111111111111')
    is_exist_bertvector = True
    if is_exist_bertvector is False:
        q_data = read_excel()
        bertconvert(q_data)
    # testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    # testQ = '上课注意力不集中怎么办？'
    testQ = '如何提高学生表达能力'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    # getBestAnswer(testQ)
    getBestAnswer2bySimilyQuestionByq2qModel(testQ)
    # getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(testQ)
