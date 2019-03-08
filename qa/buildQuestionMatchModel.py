#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import csv
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np


q_q1 = []
q_q2 = []

def papreData():
    file_dir = 'G:/tf-start/Implementation-of-Question-Answering-System/data/atec_nlp1.csv'
    # 导入18668条数据
    bc = BertClient()
    with open(file_dir, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        cou = 0
        for i in read:
            # print(type([i[0], i[1]]), [i[0], i[1]])
            tmp1 = bc.encode([i[0], i[1]])
            tmp2 = bc.encode([i[0], i[2]])
            # print(len(tmp1[0]), len(tmp1))
            # print(i[0], type(i[1]), i[2], len(i))
            q_q1.append(tmp1)
            q_q2.append(tmp2)
    print('数据导入18668条及预处理完成------------------')

    print(q_q1[0])
    print(np.append(a, [[7, 8, 9]], axis=0))


def initMLP():
    # 设置输入层节点为784，隐藏层节点为300
    in_units = 768*2
    h1_units = 300
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, 2]))
    b2 = tf.Variable(tf.zeros([2]))


    # 给训练图定义三个占位符，x，y_作为minit的train set数据接口，keep_prob作为dropout的调整接口
    x = tf.placeholder(tf.float32, [None, in_units])
    y_ = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)

    # 隐含层用ReLU作为激活函数，
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

if __name__ == '__main__':
    papreData()
