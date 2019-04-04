#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import xlrd, os
import numpy as np

path = os.path.abspath('..')
filePath = path + '/data/qa-all-data.xlsx'

'''
1默认问题答案匹配方案,通过生成的向量间余弦相似度比较,得到最相似的问题,再得到对应答案
'''


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


if __name__ == '__main__':
    # print('START------111111111111111111')
    # testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    # testQ = '上课注意力不集中怎么办？'
    testQ = '如何提高学生表达能力'
    # testQ = '学生们要面对学业上的压力，家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯'
    getBestAnswer(testQ)
