

import tensorflow as tf
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from scipy import misc
#import scipy.io as sio


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value])
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
    )


root_path = 'E://data//train//'
tfrecords_filename = root_path + 'tfrecords/car.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

txt_file = 'E://data//train//train.txt'
fr = open(txt_file)

for i in fr.readlines():
    image_path = os.path.join(root_path ,i.split()[1]+ '//' + i.split()[0])
    image_label = int(i.split()[1])
    image = np.float32((cv2.imread(image_path)))
    #image = np.float64(misc.imresize(cv2.imread(image_path), (384, 384, 3)))
    #mask = np.float64(misc.imresize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (384, 384)))
    image_raw = image.tostring()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image_raw': _bytes_feature(image_raw),
                'image_label': _int64_feature(image_label)

            }
        )
    )

    writer.write(example.SerializeToString())
    print(i)

writer.close()
fr.close()
