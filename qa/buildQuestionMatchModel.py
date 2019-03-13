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
        x1_train_data = [[] for i in range(18668)]
        x2_train_data = [[] for i in range(18668)]
        index = 0
        for i in read:
            # print(i[0], i[1], i[2])
            tmp0 = bc.encode([i[0]])
            tmp1 = bc.encode([i[1]])
            tmp2 = bc.encode([i[2]])
            # print(tmp0, tmp1, tmp2)

            qq1_vec = np.append(tmp0, tmp1)
            qq2_vec = np.append(tmp0, tmp2)
            # print(qq1_vec == qq2_vec)
            x1_train_data[index] = qq1_vec.tolist()
            x2_train_data[index] = qq2_vec.tolist()
            index += 1
            # print(x1_train_data[0])
            # print(x2_train_data[0])
            if index % 2000 == 0:
                print(x1_train_data == x2_train_data)
            if index >2000:
                break
        print('sssssssssssssssssssssssssss', len(x1_train_data[18667]))
        x1_train = np.array(x1_train_data)
        x2_train = np.array(x2_train_data)
        np.save("x1_train.npy", x1_train)
        np.save("x2_train.npy", x2_train)
    print('数据导入18668条及预处理完成------------------')


def getTrainData():
    x1_train = np.load("x1_train.npy")
    x2_train = np.load('x2_train.npy')
    print('数据导入完成')
    for index in range(len(c)):
        if x1_train[index] == x2_train[index]:
            print('数据x1 == x2， 可能出现问题了')
    return x1_train, x2_train


def MLPModel(x):

    '''接着初始化输入层与隐层间的权值、隐层神经元的300个bias、隐层与输出层之间的300x2个权值、
    输出层的2个bias，其中为了避免隐层的relu激活时陷入0梯度的情况，对输入层和隐层间的权值初始化为均值为0，
    标准差为0.2的正态分布随机数，对其他参数初始化为0：'''
    in_units = 1536    #定义输入层神经元个数
    h1_units = 300     #定义隐层神经元个数 待修改
    '''为输入层与隐层神经元之间的连接权重初始化持久的正态分布随机数，这里权重为1536乘300，300是隐层的尺寸'''
    W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], mean=0, stddev=0.2))
    b1 = tf.Variable(tf.zeros([h1_units]))
    W2 = tf.Variable(tf.zeros([h1_units, 2]))
    b2 = tf.Variable(tf.zeros([2]))
    '''接着我们定义自变量、隐层神经元dropout中的保留比例keep_prob的输入部件：'''
    x = tf.placeholder(tf.float32, [None, in_units])   #定义自变量的输入部件，为任意行X1536列
    keep_prob = tf.placeholder(tf.float32)
    # y_ = tf.placeholder(tf.float32, [None, 2])
    '''接着定义隐层relu激活部分的计算部件、隐层dropout部分的操作部件、输出层softmax的计算部件：'''
    hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)    #定义输出层softmax计算部件'''
    print('tens', y)
    return y



def trainModel(x1):
    '''注册默认的session，之后的运算都会在这个session中进行'''
    sess = tf.InteractiveSession()
    x1 = np.load("x1_train.npy")
    x2 = np.load("x1_train.npy")
    y1 = MLPModel(x1)
    y2 = MLPModel(x2)
    # 用输出y1与y2之间的max来定义loss函数
    loss = tf.maximum(0, y2-y1)
    # 使用Adagrad作为优化器
    train_step = tf.train.AdagradOptimizer(0.3).minimize(loss)
    print(y2)
    # 初始化所有变量
    tf.global_variables_initializer().run()
    # 训练，即运行训练图
    for i in range(3000):
        batch_xs, batch_ys = x1.train.next_batch(100)
        # train_step.run({x1: batch_xs, y_: batch_ys, keep_prob: 0.75})



if __name__ == '__main__':
    if 0 == 0:
        papreData()   #只在开始时候构造一次数据即可
    X1, X2 = getTrainData()
    # trainModel(x1)

    # Parameters
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 100
    display_step = 1
    # Network Parameters
    n_input = 1536  # Number of feature
    n_hidden_1 = 300  # 1st layer number of features
    n_classes = 1  # Number of classes to predict
    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # Create model
    def multilayer_perceptron(x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Output layer with linear activation
        out_layer = tf.nn.sigmoid(tf.matmul(layer_1, weights['out']) + biases['out'])
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
    pred1 = multilayer_perceptron(X1, weights, biases)
    pred2 = multilayer_perceptron(X2, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(pred1 - pred2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).maximize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(X1) / batch_size)
            X_batches = np.array_split(X1, total_batch)
            Y_batches = np.array_split(X2, total_batch)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = X_batches[i], Y_batches[i]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        #save model
        saver = tf.train.Saver()
        saver.save(sess, 'mlpmodel')  # 保存模型
