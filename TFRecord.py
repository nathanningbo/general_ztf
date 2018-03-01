

import tensorflow as tf
import os
import cv2
import numpy as np
import re
from scipy import misc
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


#root_path = 'E://data//train//'
#tfrecords_filename = root_path + 'tfrecords/car.tfrecords'
#writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#txt_file = 'E://data//train//train.txt'
def gen_TFRecord_file( rootpath, tfrecordfilename, txt_file , imgsize, img_encode_type = np.float32):
    TFRecordPath = os.path.join( rootpath,'tfrecords')
    if (not os.path.exists( TFRecordPath)):
        os.mkdir( TFRecordPath);
    writer = tf.python_io.TFRecordWriter(os.path.join( TFRecordPath, tfrecordfilename))
    fr = open(txt_file)
    for i in fr.readlines():
        wholedata = re.split(' ', i)
        #image_path = os.path.join(rootpath ,i.split()[1]+ '//' + i.split()[0])
        image_path = os.path.join(rootpath + '//images//' + wholedata[1] + '//', wholedata[0])
        image = img_encode_type(misc.imresize(cv2.imread(image_path), (imgsize[0], imgsize[1], imgsize[2])))
        image_label = int(wholedata[1])
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
if( __name__ == '__main__'):
    gen_TFRecord_file('E://data' ,'car.tfrecords', 'E://data//images//train.txt',[32,32,3])