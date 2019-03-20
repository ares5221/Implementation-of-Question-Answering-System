#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# 读取excel表格中数据，构建全部q a1 a2数据集   及剔除重复问题的q a1 a2数据集
import csv
import random
import xlrd

filePath = r'G:/tf-start/Implementation-of-Question-Answering-System/data/qa-all-data.xlsx'


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


def get_AllQuestAns():
    print('开始从excel文档中get question...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    save_alldata = []
    ques = []
    ans = []
    wroAns = []
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
            question = ''.join(question.split())
        else:
            question = sheet.cell(i, 1).value
            question = ''.join(question.split())
        # print(i-1, question)
        ques.append(question)
        if sheet.cell(i, 13).value is not '':
            answer = sheet.cell(i, 13).value
        else:
            answer = '该问题正在讨论中。。。'
        ans.append(answer)

        ranId = random.randint(2, sheet.nrows-2)
        # print(ranId)
        if sheet.cell(ranId, 13).value is not '':
            wrongAns = sheet.cell(ranId, 13).value
        else:
            wrongAns = '该问题尚没有人回答。。。'
        wroAns.append(wrongAns)

        allQAA = [ques[i - 2], ans[i - 2], wroAns[i - 2]]
        save_alldata.append(allQAA)
        with open("AllquestAnsWroA.csv", "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(allQAA)
    print('全部的问题/答案/错误答案数据已经保存完毕')
    return save_alldata

def get_atecQuestAns():
    file_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp.csv'
    # 导入204746条数据
    saveatecdata = []
    with open(file_dir, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            if len(i) > 3:
                # print(i[1], i[2], i[3])
                allqa = [i[1], i[2], i[3]]
                saveatecdata.append(allqa)
    return saveatecdata

if __name__ == '__main__':
    isgetAllQA = True
    get_atecQuestAns()
    # if isgetAllQA:
    #     get_AllQuestAns()
    # else:
    #     questions = get_question()
    #     get_ans(questions)
