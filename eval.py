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
root_path = 'E://data//train//'
tfrecord_file = os.path.join(root_path, 'tfrecords/car.tfrecords')
filename_queue = tf.train.string_input_producer([tfrecord_file])
image, label = read_and_decode(filename_queue, 50,shuffle_batch=False)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    i=0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    while not coord.should_stop():
        a,b= sess.run([image,label])
        if(b[1]==1):
            i=i+1
        print(b)
        print(i)