import tensorflow as tf
import numpy as np
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def get_all_files(file_path, is_random=True):
    """
    Get pic dir and tags
    :param file_path: a sting, dir of pic
    :param is_random: True or False, shuffle?
    :return:
    """
    image_list = []
    label_list = []

    cat_count = 0
    dog_count = 0
    for item in os.listdir(file_path):
        item_path = file_path + '\\' + item
        item_label = item.split('.')[0]  # pole.0.jpeg

        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            raise ValueError('There are unvalid documents in train document.')
        #add labels
        if item_label == 'pole':
            label_list.append(0)
            pole_count += 1
        else:
            label_list.append(1)
            other_count += 1
    print('There are %dcats,%d dogs in training set.' % (pole_count, other_count))

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)
    # shuffle
    if is_random:
        rnd_index = np.arange(len(image_list))
        np.random.shuffle(rnd_index)
        image_list = image_list[rnd_index]
        label_list = label_list[rnd_index]

    return image_list, label_list


def get_batch(train_list, image_size, batch_size, capacity, is_random=True):
    """
    get batch
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, size of pic
    :param batch_size: a int
    :param capacity: a int, size of queue
    :param is_random: True or False, shuffle?
    :return:
    """

    intput_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    # load pic from dir
    image_train = tf.read_file(intput_queue[0])
    image_train = tf.image.decode_jpeg(image_train, channels=3)  # jpeg pic
    image_train = tf.image.resize_images(image_train, [image_size, image_size])
    image_train = tf.cast(image_train, tf.float32) / 255.  # transform pic

    # pic tags
    label_train = intput_queue[1]

    # get batch
    if is_random:
        image_train_batch, label_train_batch = tf.train.shuffle_batch([image_train, label_train],
                                                                      batch_size=batch_size,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=100,
                                                                      num_threads=2)
    else:
        image_train_batch, label_train_batch = tf.train.batch([image_train, label_train],
                                                              batch_size=1,
                                                              capacity=capacity,
                                                              num_threads=1)
    return image_train_batch, label_train_batch

# pic tag
# test
##test pic###
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     # Test pic
#     image_dir = os.getcwd() + '/data/train'
#     train_list = get_all_files(image_dir, True)
#     image_train_batch, label_train_batch = get_batch(train_list, 256, 1, 200, False)
#
#     sess = tf.Session()
#
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     try:
#         for step in range(10):
#             if coord.should_stop():
#                 break
#
#             image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
#             if label_batch[0] == 0:
#                 label = 'Cat'
#             else:
#                 label = 'Dog'
#             plt.imshow(image_batch[0]), plt.title(label)
#             plt.show()
#
#     except tf.errors.OutOfRangeError:
#         print('Done.')
#     finally:
#         coord.request_stop()
#
#     coord.join(threads=threads)
#     sess.close()


import matplotlib.pyplot as plt

BATCH_SIZE = 5
CAPACITY = 256
IMG_W = 208
IMG_H = 208

image_dir = os.getcwd() + '/data/train'

image_list = get_all_files(image_dir)
image_batch, label_batch = get_batch(image_list, IMG_W, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i<1:
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE ):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1

    except(tf.errors.OutOfRangeError):
        print('Done')
    finally:
        coord.request_stop()
    coord.join(threads)





