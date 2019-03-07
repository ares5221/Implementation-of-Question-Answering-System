#引入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

#设置输入层节点为784，隐藏层节点为300
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

#给训练图定义三个占位符，x，y_作为minit的train set数据接口，keep_prob作为dropout的调整接口
x = tf.placeholder(tf.float32, [None, in_units])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

#隐含层用ReLU作为激活函数，
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

#用输出y与标签y_之间的交叉熵来定义loss函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#使用Adagrad作为优化器
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

#初始化所有变量
tf.global_variables_initializer().run()
#训练，即运行训练图
for i in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

#测试
#先构造测试图
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#运行测试图
#给x喂测试数据，通过测试图，产生测试数据的预测值y。给y_喂测试值，与已求出的y在测试图上进行比对，打印结果。
print('Test Accuracy is',accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
