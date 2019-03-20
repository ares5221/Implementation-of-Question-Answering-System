#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import csv
from bert_serving.client import BertClient
# X, Y = make_classification(n_samples=50000, n_features=10, n_informative=8,
#                            n_redundant=0, n_clusters_per_class=2)
# Y = np.array([Y, -(Y-1)]).T  # The model currently needs one column for each class
# print(Y)
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


def getData():
    datasize = 200000
    X = [[] for i in range(datasize)]
    Y = [[] for i in range(datasize)]
    data = get_atecQuestAns()
    bc = BertClient()
    for index in range(datasize):
        tmp = data[index]
        v1 = bc.encode([tmp[0]])
        v2 = bc.encode([tmp[1]])
        qq1_vec = np.append(v1, v2)
        X[index] = qq1_vec.tolist()
        Y[index] = int(tmp[2])
        if index % 100 == 0:
            print(index, 'is finish')
            print(tmp[0], tmp[1], tmp[2])
    X_train = np.array(X)
    Y_train = np.array(Y)
    np.save("Y_atec_200000.npy", Y_train)
    np.save("X_atec_200000.npy", X_train)
    print(X_train.shape)
    print(Y_train.shape)
    print('save x train')



#Start Position----------->>>>>>>>>
if __name__ == '__main__':
    getData()
    pass


X = np.load("X_labelmark5000.npy")
Y = np.load("Y_labelmark5000.npy")

Y_label = np.array([Y, -(Y-1)]).T
print(X.shape, Y_label.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_label)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(Y_test)
# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 1
display_step = 10

# Network Parameters
n_input = 1536 # Number of feature
n_hidden_1 = 1100 # 1st layer number of features
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # out_layer = tf.nn.softmax(out_layer)
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

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)
        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(Y_train, total_batch)
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
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})
    print(result)