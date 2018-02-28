# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from read_from_TFRecord import read_and_decode
import os
import netmodel

# def dense_to_one_hot(labels_dense, num_classes=2):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = np.arange(num_labels) * num_classes
#   labels_one_hot = np.zeros((num_labels, num_classes))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot
root_path = 'D://data//train//'
tfrecord_file = os.path.join(root_path, 'tfrecords/car.tfrecords')
filename_queue = tf.train.string_input_producer([tfrecord_file])
image, label = read_and_decode(filename_queue, 5)

y_ = tf.one_hot(label,2)
prediction = netmodel.inference( image, 0.5)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.global_variables_initializer()
# Init model
with tf.Session() as sess:
    sess.run(init_op)
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # while not coord.should_stop():
    #     #sess.run(train_step, feed_dict={x: image, y_: dense_to_one_hot(mask)})
    #     _,s = sess.run([train_step,cross_entropy])
    #     print(s)
    while not coord.should_stop():
        i +=1
        #sess.run(y_)
        sess.run(train_step)
        if(i%5 == 0 ):
            print(sess.run(cross_entropy))
        if(i%100 == 0 ):
             correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
             print('----------------')
             print(sess.run(correct_prediction))
             print('----------------')
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
