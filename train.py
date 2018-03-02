# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from read_from_TFRecord import read_and_decode
import os
import netmodel2

# def dense_to_one_hot(labels_dense, num_classes=2):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = np.arange(num_labels) * num_classes
#   labels_one_hot = np.zeros((num_labels, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot
root_path = 'E://data//images//'
tfrecord_file = os.path.join(root_path, 'tfrecords//car.tfrecords')
filename_queue = tf.train.string_input_producer([tfrecord_file])
image, label = read_and_decode(filename_queue,img_decode_type = tf.float32)

y_ = tf.one_hot(label,2)
y = netmodel2.inference( image, 0.5)

cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
lr = tf.placeholder(tf.float32, shape=[])
train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

init_op = tf.global_variables_initializer()
is_train = True
#is_train = False
with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver(max_to_keep=5)
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if is_train:
        while not coord.should_stop():
            i +=1
            if i < 1000:
                lrn_rate = 0.01
            elif i < 4000:
                _lrn_rate = 0.001
            else:
                _lrn_rate = 0.0001
            sess.run(train_step,feed_dict={lr:lrn_rate})
            if(i%10 == 0 ):
                print(sess.run(cross_entropy))
            if(i%30 == 0 ):
                 # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                 # print('----------------')
                 # print(sess.run(correct_prediction))
                 # print('----------------')
                 saver.save(sess, 'E://data//images//checkpoint//car.ckpt', global_step=i)
                 print('save done')
                 print(i)
            if(i%8100 == 0):
                coord.request_stop()
    else:
        model_file = tf.train.get_checkpoint_state('E://data//images//checkpoint//')
        saver.restore(sess,'E://data//images//checkpoint'+ '.\\'+'car.ckpt-200')
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        while not coord.should_stop():
            #print(sess.run(y))
            # print(sess.run(y_))
            a =sess.run(correct_prediction)
            b =np.sum(a)/50
            print(b)
            #print(sess.run(cross_entropy))


            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(sess.run(accuracy))
#sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
# Train
# for i in range(10000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#     if(i%1000==0):
#         print(i)

# Test trained model

# print(sess.run(accuracy, feed_dict={x: image,
#                                     y_: dense_to_one_hot(mask)}))
