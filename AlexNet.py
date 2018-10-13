import cv2
import numpy as np 
import os
from random import shuffle
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

batch_size = 32
num_bathes = 100

train_data = 'C:/Users/13302/OneDrive/Documents/UtilityPoles/Train'
test_data = 'C:/Users/13302/OneDrive/Documents/UtilityPoles/Test'
#Set tag on pictures
def one_hot_lable(img):
    label = img.split('.')[0]
    if label == 'poles':
        ohl = np.array([1,0])
    elif label == 'other':
        ohl = np.array([0,1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        train_images.append([np.array(img), one_hot_lable(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        test_images.append([np.array(img), one_hot_lable(i)])
    return test_images

def __main__():
    training_images = train_data_with_label()
    testing_images = test_data_with_label()
    #Get array data of pictures
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1, 64, 64, 1)
    tr_lbl_data = np.array([i[1] for i in training_images])

    tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1, 64, 64, 1)
    tst_lbl_data = np.array([i[1] for i in testing_images]) 

    #Send data to CNN
    images = tr_img_data
    output,parameters = inference(images)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    time_tensorflow_run(sess,output,"Forward")
    objective = tf.nn.l2_loss(output)
    grad = tf.gradients(objective,parameters)
    time_tensorflow_run(sess,grad,"Forward-backward")


def inference(images):
    parameters = []

    #First CL
    with tf.name_scope("conv1") as scope:
        kernel1 = tf.Variable(tf.truncated_normal([11,11,3,64],mean=0,stddev=0.1,
                                                  dtype=tf.float32),name="weights")
        conv = tf.nn.conv2d(images,kernel1,[1,4,4,1],padding="SAME")
        biases = tf.Variable(tf.constant(0,shape=[64],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        parameters += [kernel1,biases]
        lrn1 = tf.nn.lrn(conv1,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn1")
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pool1")
        print_tensor_info(pool1)

    #Second CL
    with tf.name_scope("conv2") as scope:
        kernel2 = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[192])
                             ,trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name=scope)
        print_tensor_info(conv2)
        parameters += [kernel2,biases]
        lrn2 = tf.nn.lrn(conv2,4,1.0,alpha=1e-3/9,beta=0.75,name="lrn2")
        pool2 = tf.nn.max_pool(lrn2,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool2")
        print_tensor_info(pool2)

    #Third CL
    with tf.name_scope("conv3") as scope:
        kernel3 = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool2,kernel3,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel3,biases]
        print_tensor_info(conv3)

    #Forth CL
    with tf.name_scope("conv4") as scope:
        kernel4 = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.1,dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv3,kernel4,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel4,biases]
        print_tensor_info(conv4)

    #Fifth CL
    with tf.name_scope("conv5") as scope:
        kernel5 = tf.Variable(tf.truncated_normal([3,3,256,256],stddev=0.1,dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv4,kernel5,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name="biases")
        bias = tf.nn.bias_add(conv,biases)
        conv5 = tf.nn.relu(bias)
        parameters += [kernel5,bias]
        pool5 = tf.nn.max_pool(conv5,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool5")
        print_tensor_info(pool5)

    #First ACL
    pool5 = tf.reshape(pool5,(-1,6*6*256))
    weight6 = tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.1,dtype=tf.float32),
                           name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5,weight6),ful_bias1))

    #Second ACL
    weight7 = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1,dtype=tf.float32),
                          name="weight7")
    ful_bias2 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[4096]),name="ful_bias2")
    ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1,weight7),ful_bias2))

    #Third ACL
    weight8 = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1,dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con2,weight8),ful_bias3))

    #Softmax
    weight9 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1),dtype=tf.float32,name="weight9")
    bias9 = tf.Variable(tf.constant(0.0,shape=[10]),dtype=tf.float32,name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3,weight9)+bias9)

    return output_softmax,parameters


if __name__ == "__main__":
    __main__()