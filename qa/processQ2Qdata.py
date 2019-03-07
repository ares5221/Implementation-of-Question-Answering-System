#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import random

def build_qdata():
    file_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp_sim_train.csv'
    save_atec_nlp_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp2.csv'
    with open(file_dir, "r") as csvfile:
        # 读取csv文件，返回的是迭代类型
        read = csv.reader(csvfile)
        for i in read:
            # print(i, len(i))
            if len(i) > 0:
                ss = i[0].split('\t')
                ss = ss[0: 4]
                print(ss, ss[0], len(ss))
                if ss[3].isdigit():
                    # get q2
                    if int(ss[3]) == 0:
                        q2 = ss[2]
        for i in read:
            if len(i)>0:
                pass


def ss():
    ranId = random.randint(0, 63130)
    while ranId - int(ss[0]) == 0:
        ranId = random.randint(0, 63130)

    with open(save_atec_nlp_dir, 'a', newline='', encoding='utf-8') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(ss)

if __name__ == '__main__':
    # step1 构造atec_nlp数据集，格式为id， q， q1，label
    build_qdata()
    # step2 构造q-q1-q2数据集，格式为q，q1， q2
