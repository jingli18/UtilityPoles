from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Basic Data Configurations
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
learning_rate = 0.001
training_iters = 200000
display_step = 20
n_classes = 10
batch_size = 128
_dropout1 = 0.85
_dropout2 = 0.5
n_input = 784

keep_prob = tf.placeholder('float')
with tf.name_scope('Inputs'):
    x = tf.placeholder('float', [None, 784], name='x')
    y = tf.placeholder('float', [None, n_classes], name='y')

def conv2d(x, weights, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME'), b))

def maxpool2d(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def lrn(x):
    return tf.nn.lrn(x, 4, bias=1, alpha=1e-3/9, beta=0.75, name="lrn")

def convolutional_neural_network(x):
    weights = {
        'W_conv1': tf.Variable(tf.random_normal([7, 7, 1, 64])),
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
        'W_conv3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
        'W_conv4': tf.Variable(tf.random_normal([3, 3, 384, 128])),
        'W_conv5': tf.Variable(tf.random_normal([3, 3, 128, 256])),

        'W_fc1': tf.Variable(tf.random_normal([4*4*192, 4096])),
        'W_fc2': tf.Variable(tf.random_normal([4096, 4096])),
        'W_fc3': tf.Variable(tf.random_normal([4096, 1000])),
        'out': tf.Variable(tf.random_normal([1000, n_classes]))}

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([64])),
        'b_conv2': tf.Variable(tf.random_normal([192])),
        'b_conv3': tf.Variable(tf.random_normal([384])),
        'b_conv4': tf.Variable(tf.random_normal([128])),
        'b_conv5': tf.Variable(tf.random_normal([256])),
        'b_fc1': tf.Variable(tf.random_normal([4096])),
        'b_fc2': tf.Variable(tf.random_normal([4096])),
        'b_fc3': tf.Variable(tf.random_normal([1000])),
        'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

#CL
    with tf.name_scope('Conv1'):
        conv1 = conv2d(x, weights['W_conv1'], biases['b_conv1'])
        conv1 = maxpool2d(conv1, k=2)
        conv1 = lrn(conv1)
        conv1 = tf.nn.dropout(conv1, _dropout1)
    with tf.name_scope('Conv2'):
        conv2 = conv2d(conv1, weights['W_conv2'], biases['b_conv2'])
        conv2 = maxpool2d(conv2, k=4)
        conv2 = lrn(conv2)
        conv2 = tf.nn.dropout(conv2, _dropout2)
    # with tf.name_scope('Conv3'):
    #     conv3 = conv2d(conv2, weights['W_conv3'], biases['b_conv3'])
    #     conv3 = maxpool2d(conv3, k=2)
    #     conv3 = lrn(conv3)
    #     conv3 = tf.nn.dropout(conv3, _dropout2)

    # with tf.name_scope('Conv4'):
    #     conv4 = conv2d(conv3, weights['W_conv4'], biases['b_conv4'])
    #     conv4 = maxpool2d(conv4, k=1)
    # with tf.name_scope('Conv5'):
    #     conv5 = conv2d(conv4, weights['W_conv5'], biases['b_conv5'])
    #     conv5 = maxpool2d(conv5, k=1)

    #FCL
    with tf.name_scope('FullConnect1'):
        fc1 = tf.reshape(conv2,[-1, weights['W_fc1'].get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])

    with tf.name_scope('FullConnect2'):
        fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])

    with tf.name_scope('FullConnect3'):
        fc3 = tf.nn.relu(tf.matmul(fc2, weights['W_fc3'])+biases['b_fc3'])

    with tf.name_scope('OutPut'):
        output = tf.matmul(fc3, weights['out'])+biases['out']
        return output




pred = convolutional_neural_network(x)
prediction = tf.nn.softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys, keep_prob: 1})
    return result

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)





for i in range(4000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    sess.run(opt, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
