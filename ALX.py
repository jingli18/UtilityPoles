import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128
_dropout1 = 0.8
_dropout2 = 0.6
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, n_classes])

def conv2d(x, weights, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME'), b))


def maxpool2d(x, k):
    #                        size of window
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def lrn(x):
    return tf.nn.lrn(x, 4, bias=1, alpha=1e-3/9, beta=0.75, name="lrn")

def lrn2(x):
    return tf.nn.lrn(x,4,1.0,alpha=1e-3/9,beta=0.75,name="lrn2")
def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([7, 7, 1, 64])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 64, 192])),
               'W_conv3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
               'W_conv4': tf.Variable(tf.random_normal([3, 3, 384, 128])),
               'W_conv5': tf.Variable(tf.random_normal([3, 3, 128, 256])),

               'W_fc1': tf.Variable(tf.random_normal([4*4*256, 4096])),
               'W_fc2': tf.Variable(tf.random_normal([4096, 4096])),
               'W_fc3': tf.Variable(tf.random_normal([4096, 1000])),
               'out': tf.Variable(tf.random_normal([1000, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([64])),
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
    conv1 = conv2d(x, weights['W_conv1'], biases['b_conv1'])
    conv1 = maxpool2d(conv1, k=2)
    conv1 = lrn(conv1)
    conv1 = tf.nn.dropout(conv1, _dropout1)

    conv2 = conv2d(conv1, weights['W_conv2'], biases['b_conv2'])
    conv2 = maxpool2d(conv2, k=2)
    conv2 = lrn(conv2)
    conv2 = tf.nn.dropout(conv2, _dropout2)

    conv3 = conv2d(conv2, weights['W_conv3'], biases['b_conv3'])
    conv3 = maxpool2d(conv3, k=2)
    conv3 = lrn(conv3)
    conv3 = tf.nn.dropout(conv3, _dropout2)


    conv4 = conv2d(conv3, weights['W_conv4'], biases['b_conv4'])
    conv4 = maxpool2d(conv4, k=1)

    conv5 = conv2d(conv4, weights['W_conv5'], biases['b_conv5'])
    conv5 = maxpool2d(conv5, k=1)

#FCL
    fc1 = tf.reshape(conv5,[-1, weights['W_fc1'].get_shape().as_list()[0]])

    fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1'])+biases['b_fc1'])

    fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2'])+biases['b_fc2'])

    fc3 = tf.nn.relu(tf.matmul(fc2, weights['W_fc3'])+biases['b_fc3'])

    output = tf.matmul(fc3, weights['out'])+biases['out']
    return output



def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)