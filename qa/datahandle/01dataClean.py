#!/usr/bin/env python
# _*_ coding:utf-8 _*_

# 数据清理,为了提高推荐答案的质量,清除错误答案及无效问题
import csv
import re
import xlrd

filePath = r'G:/tf-start/Implementation-of-Question-Answering-System/data/qa-all-data.xlsx'
qa_cleandata_dir = r'G:/tf-start/Implementation-of-Question-Answering-System/data/qa-clean-data.csv'


def manage_question(sheet):
    for i in range(2, sheet.nrows):
        print(i)
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
            question = "".join(question.split())
            question = re.sub(r'^第.*周问题：', "", question)
            question = re.sub(r'^.*周问题：', "", question)
        else:
            question = sheet.cell(i, 1).value
            question = "".join(question.split())
            question = re.sub(r'^第.*周问题：', "", question)
            question = re.sub(r'^.*周问题：', "", question)

        if sheet.cell(i, 13).value is not '':
            answer = sheet.cell(i, 13).value
            answer = "".join(answer.split())
            if len(answer) < 4:
                continue
            if re.match('老师您好', answer):
                continue
            if re.match('正在讨论', answer):
                continue
        else:
            continue
        print(question, answer)
        news = [question, answer]
        with open(qa_cleandata_dir, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
        print('保存在csv文件中!!!!!')


if __name__ == '__main__':
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)

    manage_question(sheet)
    print('data clean success!!!!!!!!!!!!!!!!')
