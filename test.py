# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from read_from_TFRecord import read_and_decode
import os
import netmodel
root_path = 'E://data//images//'
tfrecord_file = os.path.join(root_path, 'tfrecords/car.tfrecords')
filename_queue = tf.train.string_input_producer([tfrecord_file])
image, label = read_and_decode(filename_queue,img_decode_type = tf.float32,shuffle_batch=False)
y_ = tf.one_hot(label,2)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    i=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while not coord.should_stop():
        a,b,c= sess.run([image,label,y_])
        if(b[1]==1):
            i=i+1
        print(i)
        print(b)
        print(c)

