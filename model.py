#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import tensorflow as tf


image_size = 227
lr = 1e-4
epoch = 200
batch_size = 50
display_step = 5
num_class = 2
num_fc1 = 4096
num_fc2 = 2048



W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
    'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
    'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
    'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
    'fc1': tf.Variable(tf.truncated_normal([5 * 5 * 256, num_fc1], stddev=0.1)),
    'fc2': tf.Variable(tf.truncated_normal([num_fc1, num_fc2], stddev=0.1)),
    'fc3': tf.Variable(tf.truncated_normal([num_fc2, num_class], stddev=0.1)),

}

b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
    'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_fc1])),
    'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_fc2])),
    'fc3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[num_class])),
}


def model(x_image):
    conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print('pool.shape{})'.format(pool1.shape)) (?, 27, 27, 96)

    conv2 = tf.nn.conv2d(pool1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print('poo2.shape{})'.format(pool2.shape)) (?, 13, 13, 256)

    conv3 = tf.nn.conv2d(pool2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    conv3 = tf.nn.relu(conv3)
    pool3 = tf.nn.avg_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID')
    # print('poo3.shape{})'.format(pool3.shape)) (?, 11, 11, 384)

    conv4 = tf.nn.conv2d(pool3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
    conv4 = tf.nn.relu(conv4)
    # print('conv4.shape{})'.format(conv4.shape)) (?, 11, 11, 384)

    conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
    conv5 = tf.nn.relu(conv5)
    pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print(pool5.shape)   (?, 5, 5, 256)

    reshaped = tf.reshape(pool5, (-1, 5 * 5 * 256))

    fc1 = tf.add(tf.matmul(reshaped, W_conv['fc1']), b_conv['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)

    fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, 0.5)

    fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

    return fc3



