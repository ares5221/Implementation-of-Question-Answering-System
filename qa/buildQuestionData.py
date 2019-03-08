#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
import random
'''
数据文件介绍：atec_nlp_sim_train.csv是源文件不要动
atec_nlp1.csv是q，q1，q2数据形式
'''
def build_qdata():
    file_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp_sim_train.csv'
    save_atec_nlp_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp1.csv'
    q2 = []
    with open(file_dir, "r") as csvfile:
        # 读取csv文件，返回的是迭代类型
        read = csv.reader(csvfile)
        count = 0
        for i in read:
            # print(i, len(i))
            if len(i) > 0:
                ss = i[0].split('\t')
                # ss = ss[0: 4]
                # print(ss, ss[0], len(ss))
                if ss[3].isdigit():
                    # get q2
                    if int(ss[3]) == 0:
                        q2.append(ss[2])
            count += 1
        print(count)
    q2len = len(q2)
    print(q2len ,q2[0],'-----------------------------')
    with open(file_dir, "r") as csvfile1:
        read1 = csv.reader(csvfile1)
        for i in read1:
            if len(i)>0:
                ss = i[0].split('\t')
                ss = ss[1: 4]
                # print(ss, ss[1], len(ss))
                if ss[2].isdigit():
                    if int(ss[2]) == 1:
                        ranId = random.randint(0, q2len-1)
                        news = [ss[0], ss[1], q2[ranId]]
                        # print(news)
                        with open(save_atec_nlp_dir, 'a', newline='', encoding='utf-8') as csvfile:
                            spamwriter = csv.writer(csvfile)
                            spamwriter.writerow(news)


if __name__ == '__main__':
    # step1 构造atec_nlp1.csv数据集，格式为q， q1，q2
    build_qdata()

