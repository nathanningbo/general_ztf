#import scipy.misc as misc
import tensorflow as tf
import numpy as np
#import scipy.io as sio
import matplotlib.pyplot as plt
import os
image_shape = 32


root_path = 'E://data//train//'
tfrecord_file = os.path.join(root_path, 'tfrecords/car.tfrecords')


def read_and_decode(filename_queue, batch_size, random_crop=False, random_filp=True, shuffle_batch=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_label': tf.FixedLenFeature([], tf.int64)
        }
    )

    image = tf.decode_raw(features['image_raw'], tf.float32)
    #image = tf.reshape(image, [image_shape*image_shape*3])
    image = tf.reshape(image, [image_shape,image_shape,3])

    #image = tf.image.per_image_standardization(image)

    label = tf.cast(features['image_label'], tf.int64)
    #label = tf.decode_raw(features['image_label'], tf.int64)
    print(image, label)
    # if random_filp:
    #     image = tf.image.random_flip_left_right(image)
    #     mask = tf.image.random_flip_left_right(mask)

    if shuffle_batch:
        image_batch, mask_batch = tf.train.shuffle_batch([image, label],
                                                         batch_size=batch_size,
                                                         capacity=batch_size*5,
                                                         num_threads=2,
                                                         min_after_dequeue=batch_size*2)
    else:
        image_batch, mask_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            capacity=3*batch_size,
            num_threads=2
        )

    return image_batch, mask_batch


def test_run(tfrecord_filename):
    filename_queue = tf.train.string_input_producer([tfrecord_filename])
    image, mask = read_and_decode(filename_queue,5)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        i= 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # for i in range(1):
        #     img, msk = sess.run([image, mask])
        #     print(img, msk)
        #
        # coord.request_stop()
        # coord.join(threads)
        try:
            while not coord.should_stop() and i < 5:
                with tf.device('/CPU:0'):
                    img, label = sess.run([image, mask])
                    print(np.shape(img))
                    print(np.shape(label))
                    print(label)

                for j in range(5):
                    plt.imshow(img[j, :, :, :])
                    #plt.imshow(img[j, :])
                    print(label)
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()

        coord.join(threads)

if( __name__ == '__main__'):
    test_run(tfrecord_filename=tfrecord_file)