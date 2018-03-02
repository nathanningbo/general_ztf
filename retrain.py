# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from read_from_TFRecord import read_and_decode
import os
import netmodel

root_path = 'E://data//images//'
tfrecord_file = os.path.join(root_path, 'tfrecords//car.tfrecords')
filename_queue = tf.train.string_input_producer([tfrecord_file])
image, label = read_and_decode(filename_queue,img_decode_type = tf.float32)

y_ = tf.one_hot(label,2)
y = netmodel.inference( image, 0.5)

cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    saver = tf.train.Saver(max_to_keep=5)
    model_file = tf.train.get_checkpoint_state('E://data//images//checkpoint//')
    saver.restore(sess, 'E://data//validate//checkpoint' + '.\\' + 'car.ckpt-100')
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while not coord.should_stop():
        i +=1
        sess.run(train_step)
        if(i%5 == 0 ):
            print(sess.run(cross_entropy))
        if(i%100 == 0 ):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            print('----------------')
            print(sess.run(correct_prediction))
            print('----------------')
            saver.save(sess, 'E://data//images//checkpoint//car.ckpt', global_step=i)

