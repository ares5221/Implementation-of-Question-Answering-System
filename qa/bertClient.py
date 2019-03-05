#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import xlrd, json
import numpy as np


filePath = r'G:/tf-start/Implementation-of-Question-Answering-System/data/qa-all-data.xlsx'
qusetionvector = []
# 1read excel
def read_excel():
    print('step1：开始从excel文档中read data...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    q_data = []  # 存放excel中获取的question 问题标题
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
        else:
            question = sheet.cell(i, 1).value
        ansName = sheet.cell(i, 9)  # 存放excel中的 回答者name
        q_data.append(question)
        # print(question, ansName)
    return q_data

def bertconvert(datas):
    q_vector = []
    question_list = []
    bc = BertClient()
    print(len(datas))

    for i in range(0, len(datas)):
        data = datas[i]
        newdata = "".join(data.split())  # data.replace(' ', '')
        # print('222get question string', newdata)
        question_list.append(newdata)
    # print(question_list)
    ss = bc.encode(question_list)
    np.save("G:/tf-start/Implementation-of-Question-Answering-System/data/question2vec1.npy", ss)


def testBERTEncode():
    bc = BertClient()
    # print(bc.encode(['我 喜欢 你们', '我 喜 欢 你 们', '我喜欢你们']))  # 模型不需要分词，发现这几种情况得到的编码向量是一样的
    s1 = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没...'
    s2 = '学生上课注意力不集中怎么办？'
    bcs1 = bc.encode([s1])
    bcs2 = bc.encode([s2])
    print(bcs1)
    print('---------test encode----------')
    print(bcs2)
    print(len(bcs1[0]), 'and ', len(bcs2[0]))


def getBestAnswer(qdata):
    b = np.load("G:/tf-start/Implementation-of-Question-Answering-System/data/question2vec1.npy")
    print(len(b), 'step 3:load question vector success!!!!!!!!!!!!!!!!')
    bc = BertClient()
    testvec = bc.encode(["".join(qdata.split())])
    print('test vector', len(testvec[0]))
    print('step4: cal cosine_similarity')
    maxsimil = 0
    for i in range(len(b)): #3625
        simil_test_ques = cosine_similarity(b[i], testvec[0])
        # print(simil_test_ques)
        if simil_test_ques > maxsimil:
            maxsimil = simil_test_ques
            index = i
    print('step4 获取到最相似问题的索引@@@@@@@@', maxsimil, index)
    index = index + 2  # 由于表中数据索引从第二列开始
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
        ans = '没有找到该问题的对应答案'
    print('step5 Finsh!!!!!!!!!')
    return ans

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
        return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)

if __name__ == '__main__':
    print('START------111111111111111111')
    is_exist_bertvector = True
    if is_exist_bertvector is False:
        q_data = read_excel()
        bertconvert(q_data)
    # testBERTEncode()  #test
    testQ = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？！如何避免讨论后小组派代表没人愿意说？或者一讨论学生们就聊别的这一问题呢？'
    # testQ = '学生上课注意力不集中怎么办？'
    testQ = '注意培养学生用多种策'
    getBestAnswer(testQ)


