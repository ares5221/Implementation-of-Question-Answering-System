#!/usr/bin/env python
# _*_ coding:utf-8 _*_

'''
清理数据 获得[ [question1,[a1[],a2[]]],
                [question2,[a1[],a2[]]] ]

'''
import os
import csv
import re
import xlrd

path = os.path.abspath('..')
filePath = path + '/data/qa-all-data.xlsx'
q_ansall_list_data_dir = path + '/data/q-a-all-list-data.csv'
qa_final_data_dir = path + '/data/qa-final-data.csv'
print(path)


def manage_question(sheet):
    '''获取到所有问题的有序无重复列表'''
    que_set = set()  # 保存不重复的所有问题集合
    questions = []   # 保存问题集合顺序列表
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 2).value is not '':
            question = sheet.cell(i, 2).value
            # question = "".join(question.split())
        #     question = re.sub(r'^第.*周问题：', "", question)
        #     question = re.sub(r'^.*周问题：', "", question)
        else:
            question = sheet.cell(i, 1).value
            # question = "".join(question.split())
        #     question = re.sub(r'^第.*周问题：', "", question)
        #     question = re.sub(r'^.*周问题：', "", question)

        if question not in que_set:
            que_set.add(question)
    # print(len(que_set), que_set)

    for j in range(2, sheet.nrows):
        if sheet.cell(j, 2).value is not '':
            question = sheet.cell(j, 2).value
            # question = "".join(question.split())
            # question = re.sub(r'^第.*周问题：', "", question)
            # question = re.sub(r'^.*周问题：', "", question)
        else:
            question = sheet.cell(j, 1).value
            # question = "".join(question.split())
            # question = re.sub(r'^第.*周问题：', "", question)
            # question = re.sub(r'^.*周问题：', "", question)

        if question in que_set:
            questions.append(question)
            que_set.remove(question)

    print(len(questions), questions)
    print('获取到所有问题的有序无重复列表 success')
    return questions


def manage_anslist(sheet, questions):
    '''获取到所有问题及对应答案列表信息'''
    que_anslist = [['', []] for i in range(len(questions))]
    for i in range(len(questions)):
        for j in range(2, sheet.nrows):
            if sheet.cell(j, 2).value is not '':
                if sheet.cell(j, 2).value == questions[i]:
                    que_anslist[i][0] = sheet.cell(j, 2).value
                    que_anslist[i][1].append(get_ans_info(j))
            else:
                if sheet.cell(j, 1).value == questions[i]:
                    que_anslist[i][0] = sheet.cell(j, 1).value
                    que_anslist[i][1].append(get_ans_info(j))

    print(len(que_anslist),'@@@@@@@@@@', que_anslist[1])
    print('获取到所有问题及对应答案列表信息  success')
    return que_anslist


def ans_select(qa_list):
    '''答案选择，从答案列表中最终选出一个答案
    策略：先比较点赞数，选最高的，若相同选最长的答案：
    todo 优先时间最晚的，用户单位会设置优先级，如优先选好老师平台
    '''
    que_ans_pair = [['', ''] for i in range(len(questions))]
    if len(qa_list) < 1:
        print('数据传输有误，查看数据')
    for i in range(len(qa_list)):
        que_ans_pair[i][0] = qa_list[i][0]

        anslist = qa_list[i][1]  # 每个问题的所有答案列表

        # 1比较点赞数
        max_thumbup_num = 0
        ans_index = 0
        for j in range(len(anslist)):
            thumbup_num_str = anslist[j][3]
            if thumbup_num_str == '':
                thumbup_num_str = '0'
            thumbup_num_int = int(thumbup_num_str)

            if thumbup_num_int > max_thumbup_num:
                max_thumbup_num= thumbup_num_int
                ans_index = j
        if max_thumbup_num > 0:
            que_ans_pair[i][1] = anslist[ans_index][2]
        # 2 比较答案长度
        if max_thumbup_num == 0:
            max_ans_length = 0
            ans__length_index = 0
            for k in range(len(anslist)):
                ans_length = len(anslist[j][2])
                if ans_length > max_ans_length:
                    max_ans_length = ans_length
                    ans__length_index = k
            que_ans_pair[i][1] = anslist[ans__length_index][2]
        # print(i,'%%%%%%%%%%%%',que_ans_pair[i][1])
    # print(len(que_ans_pair), que_ans_pair)
    return que_ans_pair





def get_ans_info(index):
    anslist = []
    # k is index of 回答用户单位/回答时间/回答内容/点赞数 in qa-all-data.xlsx
    for k in range(11, 15):
        tmp = sheet.cell(index, k).value
        anslist.append(tmp)
    # print(anslist)
    return anslist

    # if sheet.cell(i, 13).value is not '':
    #     answer = sheet.cell(i, 13).value
    #     answer = "".join(answer.split())
    #     if len(answer) < 4:
    #         continue
    #     if re.match('老师您好', answer):
    #         continue
    #     if re.match('正在讨论', answer):
    #         continue
    # else:
    #     continue

def save_qaalllist(qa):
    for news in qa:
        with open(q_ansall_list_data_dir, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
    print('保存在csv文件中!!!!!')


def save_finalQAdata(qa):
    for news in qa:
        with open(qa_final_data_dir, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
    print('处理后的问答对数据保存在qa_final_data.csv文件中!!!!!')



if __name__ == '__main__':
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)

    questions = manage_question(sheet)
    qa_list = manage_anslist(sheet, questions)
    if not os.path.exists(q_ansall_list_data_dir):  # 如果不存在
        save_qaalllist(qa_list)
    finalQAdata = ans_select(qa_list)
    if not os.path.exists(qa_final_data_dir):  # 如果不存在
        save_finalQAdata(finalQAdata)

    # todo问题需要question = "".join(question.split())
    #         #     question = re.sub(r'^第.*周问题：', "", question)
    # 将新的问题答案数据转换为向量保存


    print('data clean success!!!!!!!!!!!!!!!!')
