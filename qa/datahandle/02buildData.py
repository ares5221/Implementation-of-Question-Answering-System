#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# 通过清理后的qa-clean-data.csv 构建question answer wronganswer数据集
import csv
import random
import xlrd
import os

path = os.path.abspath('../..')
filePath = path + '/data/qa-clean-data.csv'


def get_question():
    print('开始从excel文档中get question...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)

    q_data = []
    q_set = set()
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
            question = ''.join(question.split())
        else:
            question = sheet.cell(i, 1).value
            question = ''.join(question.split())
        # print(i-1, question)
        if sheet.cell(i, 13).value is not '':
            if question not in q_set:
                q_set.add(question)
                q_data.append(question)
                # print(question)
    return q_data

#useless code
def get_ans(data):
    print(len(data))
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    answers = ['' for i in range(636)]
    for i in range(len(data)):
        count = 0
        ans = ''
        for j in range(2, sheet.nrows):
            qq = ''.join(sheet.cell(j, 2).value.split())
            if qq == '':
                qq = ''.join(sheet.cell(j, 1).value.split())
            if data[i] == qq:
                answer = sheet.cell(j, 13).value
                count += 1
                if len(answer) > len(ans):
                    ans = answer
                    answers[i] = ans
        print('sss', data[i], '---------------------', answers[i])
        wrongAns = '该问题正在讨论中。。。。'
        with open("questAns.csv", "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            ww = [data[i], answers[i], wrongAns]
            writer.writerow(ww)


def get_AllquestAnsWroA():
    # 收集所有答案便于选取任意答案作为错误答案
    allAns = []
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            # print(i[0], i[1])
            allAns.append(i[1])
    # 保存为csv
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            print(i[0], i[1])
            ranIndex = random.randint(0, len(allAns)-1)
            wroAns = allAns[ranIndex]
            allQAA = [i[0], i[1], wroAns]
            with open(path + "/data/AllquestAnsWroA1.csv", "a", newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(allQAA)
        print('全部的问题/答案/错误答案数据已经保存完毕')


if __name__ == '__main__':
    get_AllquestAnsWroA()
    print(path)
