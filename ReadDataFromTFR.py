#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import tensorflow as tf


def read_and_decode(tfrecords_file, batch_size, image_size=227):  # 默认是AlexNet
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string)
                                           }) # 这里是一次读取一张图片
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image = tf.reshape(image, [image_size, image_size, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      min_after_dequeue=100,
                                                      num_threads=64,
                                                      capacity=200)

    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


if __name__ == '__main__':
    import os
    tfr_files = []
    for root, dirs, files in os.walk('./TFRecord'):  # 若想使用绝对路径，指定绝对路径
        for file in files:
            tfr_files.append(os.path.join(root, file))
    image_batch, label_batch = read_and_decode(tfr_files[0], 32)
    print(image_batch.shape)  # (32, 227, 227, 3)
    print(label_batch.shape)  # (32,)
