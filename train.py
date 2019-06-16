#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

# 这是之前写的文件
import GetImageLabel
import GetBatch
import ReadDataFromTFR
import model
import OneHot

image_size = 227
num_class = 2
lr = 0.001
epoch = 2000

x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
y = tf.placeholder(tf.int64, [None, num_class])


def train(image_batch, label_batch,val_Xbatch, val_ybatch):

    fc3 = model.model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(fc3, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        save_model = './model'
        save_log = './log'
        save_plt = './plt'
        max_acc = 0

        if not os.path.exists(save_model):
            print('模型保存目录{}不存在，正在创建……'.format(save_model))
            os.mkdir(save_model)
            print('创建成功')
        if not os.path.exists(save_log):
            print('日志保存目录{}不存在，正在创建……'.format(save_log))
            os.mkdir(save_log)
            print('创建成功')
        if not os.path.exists(save_plt):
            print('损失可视化保存目录{}不存在，正在创建……'.format(save_plt))
            os.mkdir(save_plt)
            print('创建成功')
        save_model += (os.sep + 'AlexNet.ckpt')
        save_plt += (os.sep + 'Alexnet.png')
        train_writer = tf.summary.FileWriter(save_log, sess.graph)
        saver = tf.train.Saver()

        losses = []
        acc = []
        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(epoch):
            image, label = sess.run([image_batch, label_batch])  # 注意 【】
            labels = OneHot.onehot(label)
            train_dict = {x: image, y: labels}

            val_image, val_label = sess.run([val_Xbatch, val_ybatch])  # 注意 【】
            val_labels = OneHot.onehot(val_label)
            val_dict = {x: val_image, y: val_labels}

            sess.run(optimizer, feed_dict=train_dict)
            loss_record = sess.run(loss, feed_dict=train_dict)
            acc_record = sess.run(accuracy, feed_dict=val_dict)

            losses.append(loss_record)
            acc.append(acc_record)
            if acc_record > max_acc:
                max_acc = acc_record

            if i % 100 == 0:
                print('正在训练，请稍后……')
                print('now the loss is {}'.format(loss_record))
                print('now the acc is {}'.format(acc_record))

                end_time = time.time()
                print('runing time is {}:'.format(end_time - start_time))
                start_time = end_time
                print('----------{} epoch is finished----------'.format(i))
                print('最大精确度为{}'.format(max_acc))
        print('训练完成，模型正在保存……')
        saver.save(sess, save_model)
        print('模型保存成功')

        coord.request_stop()
        coord.join()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(acc)
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.tight_layout()
    plt.savefig(save_plt, dpi=200)
    plt.show()


if __name__ == '__main__':
    tfrecords = './TFRecord/train/train.tfrecords'
    file_train = './resized_images/train'
    file_val = './resized_images/validation'
    # image_batch, label_batch = ReadDataFromTFR.read_and_decode(tfrecords, 32)
    X_train, y_train = GetImageLabel.get_file(file_train)
    train_Xbatch, train_ybatch = GetBatch.get_batch(X_train, y_train, 227, 64, 200)
    X_val, y_val = GetImageLabel.get_file(file_val)
    val_Xbatch, val_ybatch = GetBatch.get_batch(X_val, y_val, 227, 64, 200)
    train(train_Xbatch, train_ybatch, val_Xbatch, val_ybatch)