import tensorflow as tf
class env():
    def conv2d(x, weights, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME'), b))

    def maxpool2d(x, k):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def lrn(x):
        return tf.nn.lrn(x, 4, bias=1, alpha=1e-3/9, beta=0.75, name="lrn")
