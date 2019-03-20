#!/usr/bin/env python
# _*_ coding:utf-8 _*_

#构建q1，a1，a2形式的qa数据集
import random
import xlrd
import numpy as np
from bert_serving.client import BertClient

filePath = r'G:/tf-start/Implementation-of-Question-Answering-System/data/qa-all-data.xlsx'


def read_excel():
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)

    qaa = [[] for i in range(sheet.nrows -2)]

    # question, answer1, answer2 = '', '', ''  # 存放excel中获取的问题, 正确答案，非正确答案
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
            question = "".join(question.split())
            # print(question, type(question))
        else:
            question = sheet.cell(i, 1).value
            question = "".join(question.split())

        if sheet.cell(i, 13).value is not '':
            answer1 = sheet.cell(i, 13).value
            answer1 = "".join(answer1.split())
        else:
            answer1 = '该问题正在讨论中...'

        #  answer2要选取不是question的答案，因此在所有答案中随机取一个即可
        ranId = random.randint(0, sheet.nrows -2)
        while ranId - i == 0:
            ranId = random.randint(0, sheet.nrows -2)
        answer2 = sheet.cell(ranId, 13).value
        answer2 = "".join(answer2.split())
        if answer2 is '':
            answer2 = '相关答案正在编辑中...'

        qaa[i-2].append(question)
        qaa[i-2].append(answer1)
        qaa[i-2].append(answer2)
    return qaa


def saveQAAVector(qaadatas):
    qaa_list = [[] for i in range(len(qaadatas))]
    bc = BertClient()
    for i in range(0, len(qaadatas)):
        data = qaadatas[i]
        # print(i, data)
        qaa_list.append(data)
        ss = bc.encode(qaadatas[i])
        # print(ss)
        qaa_list[i] = ss
    np.save("G:/tf-start/Implementation-of-Question-Answering-System/data/qaa2vec.npy", qaa_list)
    print('save qaa2vec is FINSH！！！')

if __name__ == '__main__':
    qaa = read_excel()
    bc = BertClient()
    # for i in range(len(qaa)):
    #     print(i, '------', qaa[i])
    # vec = bc.encode(qaa[10])
    # print(vec)
    # print(len(vec))
    saveQAAVector(qaa)

    b = np.load("G:/tf-start/Implementation-of-Question-Answering-System/data/qaa2vec.npy")
    print('step 3:load question vector success!!!!!!!!!!!!!!!!', len(b))
    print(b[0])
    print(len(b))

