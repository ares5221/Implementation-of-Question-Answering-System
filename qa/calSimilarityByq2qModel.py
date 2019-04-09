#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np


def calSimilarityByq2qModel(q1vec, q2vec):
    learning_rate = 0.001
    batch_size = 1
    # Network Parameters
    n_input = 1536  # Number of feature
    n_hidden_1 = 32  # 1st layer number of features
    n_classes = 2  # Number of classes to predict

    x = tf.placeholder("float", [batch_size, n_input])
    y = tf.placeholder("float", [batch_size, n_classes])

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
        model_file = tf.train.latest_checkpoint('ckptque2que/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        ##########################################
        '''
        从问答数据中选取5个语义最相似的问题作为候选问题
        '''
        # maxsimil, index = 0, 0
        simlist, idlist = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        for i in range(len(q1vec)):
            testbatch = [[] for i in range(1)]
            testbatch[0] = np.append(q1vec[i], q2vec).tolist()
            res = sess.run(pred, feed_dict={x: testbatch})[0][0]
            # print(res, i)
            if res > min(simlist):
                simlist[simlist.index(min(simlist))] = res
                idlist[simlist.index(min(simlist))] = i
            #     print(res, i)
            # if i == 0:
            #     print(q1vec[0])
            #     print(q2vec)
            # if res > maxsimil:
            #     maxsimil = res
            #     index = i
            #     print(index, maxsimil)
            # if res >= 0.95:
            #     maxsimil = res
            #     index = i
            #     print(index, maxsimil)
            #     break
        print('问题相似度:', simlist)
        print('相似问题ID:', idlist)
    #############################################
    # return index, maxsimil
    return idlist, simlist


if __name__ == '__main__':
    s1 = ['你好', '测试测试', '上课注意力不集中怎么办？', '不爱学习', '不喜欢上课']
    s2 = '你好'
    bc = BertClient()
    source = bc.encode(s1)
    query = bc.encode([s2])
    res = calSimilarityByq2qModel(source, query)
    print(res)  # output is (0, 1.0)
