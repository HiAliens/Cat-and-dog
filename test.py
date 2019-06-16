#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import tensorflow as tf
from PIL import Image
import numpy as np

import model


def per_calss(imagefile):
    image = Image.open(imagefile)
    image = image.resize([227, 227])
    image_array = np.array(image)

    image = tf.cast(image_array, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 227, 227, 3])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('./model')
        saver.restore(sess, save_model)
        image = sess.run(image)
        image_size = 227
        x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        fc3 = model.model(x)
        prediction = sess.run(fc3, feed_dict={x : image})

        max_index = np.argmax(prediction)
        if max_index == 0:
            return 'cat'
        else:
            return 'dog'


if __name__ == '__main__':

    print(per_calss('./resized_images/test/cats/cat.1512.jpg'))