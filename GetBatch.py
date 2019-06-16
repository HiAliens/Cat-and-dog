#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang

'''
ReadDataFromTFR 是将图片数据转化为TFR，当数据量庞大时，转换时间过长，这个办法是将图片地址转化为TFR，在训练过程中，
根据地址读取图片数据
'''

import tensorflow as tf


def get_batch(image_list, label_list, img_size, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
    image = tf.image.per_image_standardization(image)  # 图片标准化
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64,capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


if __name__ == '__main__':
    import GetImageLabel
    import os
    cwd = os.getcwd()
    path = os.path.join(cwd, 'resized_images')
    dirs = os.listdir('./resized_images')
    image_list, label_list = [], []
    image_list, label_list = GetImageLabel.get_file(path)
    with tf.Session() as sess:
        image, label = get_batch(image_list, label_list, 227, 32, 200)
        image2, label2 = sess.run([image, label])
        print(label2)