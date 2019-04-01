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
        # model_file = tf.train.latest_checkpoint('ckptdisplay/')
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

##########################################
        maxsimil, index = 0, 0
        for i in range(len(q1vec)):  # 3625
            testbatch = [[] for i in range(1)]
            testbatch[0] = np.append(q1vec[i], q2vec).tolist()
            # print('s1 s2相似度为', sess.run(pred, feed_dict={x: testbatch}))
            res = sess.run(pred, feed_dict={x: testbatch})[0][0]
            if res > maxsimil:
                maxsimil = res
                index = i
                print(index, maxsimil)
            # if res >= 0.95:
            #     maxsimil = res
            #     index = i
            #     print(index, maxsimil)
            #     break
#############################################
        # testbatch = [[] for i in range(1)]
        # testbatch[0] = np.append(q1vec, q2vec).tolist()
        # # print('s1 s2相似度为', sess.run(pred, feed_dict={x: testbatch}))
        # res = sess.run(pred, feed_dict={x: testbatch})[0][0]
    return index, maxsimil



if __name__ == '__main__':
    s1 = ['你好', '这是测试', '上课注意力不集中怎么办？']
    s2 = '你好'
    bc = BertClient()
    source = bc.encode(s1)
    query = bc.encode([s2])
    print(query)
    res = calSimilarityByq2qModel(source, query)
    print(res)  # output is (0, 1.0)




