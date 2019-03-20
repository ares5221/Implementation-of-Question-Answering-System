#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np
import os, xlrd

path = os.path.abspath('..')
filePath = path + '/data/qa-all-data.xlsx'


def read_excel():
    print('开始从excel文档中read data...')
    workbook = xlrd.open_workbook(filePath)
    sheet_name = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheet_name)
    a_data = []  # 存放excel中获取的question 问题标题
    for i in range(2, sheet.nrows):
        if sheet.cell(i, 13).value is not '':
            ans = sheet.cell(i, 13).value
        else:
            ans = '相关答案正在编辑中...'
        a_data.append(ans)
    return a_data


def getAnsVector(datas):
    ans_list = []
    bc = BertClient()
    print(len(datas))
    for i in range(0, len(datas)):
        data = datas[i]
        newdata = "".join(data.split())  # data.replace(' ', '')
        ans_list.append(newdata)
    ss = bc.encode(ans_list)
    np.save(path + "/data/answer2vector.npy", ss)
    print('qa-all-data文件中答案通过bert转换为vector')


def getAnswerByq2aModel(q1vec, q2vec):
    learning_rate = 0.001
    batch_size = 100
    # Network Parameters
    n_input = 1536  # Number of feature
    n_hidden_1 = 32  # 1st layer number of features
    n_classes = 2  # Number of classes to predict

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Create model
    def multilayer_perceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        out_layer = tf.nn.softmax(out_layer)
        return out_layer

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model_file = tf.train.latest_checkpoint('ckptque2ans/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)
        ##########################################
        maxsimil, index = 0, 0
        for i in range(len(q1vec)):  # 3625
            testbatch = [[] for i in range(1)]
            testbatch[0] = np.append(q1vec[i], q2vec).tolist()
            res = sess.run(pred, feed_dict={x: testbatch})[0][0]
            # if res > maxsimil:
            #     maxsimil = res
            #     index = i
            #     print(index, maxsimil)
            if res > 0.99:
                maxsimil = res
                index = i
                print(index, maxsimil)
                break
    #############################################
    # testbatch = [[] for i in range(1)]
    # testbatch[0] = np.append(q1vec, q2vec).tolist()
    # # print('s1 s2相似度为', sess.run(pred, feed_dict={x: testbatch}))
    # res = sess.run(pred, feed_dict={x: testbatch})[0][0]
    return index, maxsimil


if __name__ == '__main__':
    is_exist_bertvector = True
    if is_exist_bertvector is False:
        q_data = read_excel()
        getAnsVector(q_data)
    source = np.load(path + "/data/answer2vector.npy")
    s2 = '老师们，我在一线的时候总有一个问题，如何能够提高小组讨论的有效性？'
    bc = BertClient()
    query = bc.encode([s2])
    res = getAnswerByq2aModel(source, query)
    print(res)  # output is (0, 1.0)
