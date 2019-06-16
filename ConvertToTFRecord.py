#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import tensorflow as tf
import os
from skimage import io


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_list, label_list, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecords') # 路径
    num_sample = len(images_list)
    project_name = save_dir.split('\\')[5]  # 具体看个人路径中项目名称在第几位置上
    writer = tf.python_io.TFRecordWriter(filename)
    print(f'{project_name}的{name}数据集转换开始……')
    for i in range(num_sample):
        try:
            image = io.imread(images_list[i])
            # print(type(image)) # must be a array
            image_raw = image.tostring()
            label = int(label_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label':int64_feature(label),
                'image_raw':bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print(f'读取{images_list[i]}失败')
    writer.close()
    print(f'{project_name}的{name}数据集转换为{name}.tfrecords完成')


if __name__ == '__main__':
    import GetImageLabel as getdata
    import pysnooper

    # @pysnooper.snoop()
    def run():
        cwd = os.getcwd()
        dirs = os.listdir('./resized_images')
        save_dir = []
        src = os.path.join(cwd, 'resized_images')
        print(src)
        for dir in dirs:
            save_dir.append(os.path.join(cwd, 'TFRecord\\' + dir))
            src += os.sep + dir
        for i in range(len(dirs)):
            if not os.path.exists('./TFRecord'):
                os.mkdir('./TFRecord')
            if not os.path.exists(save_dir[i]):
                os.mkdir(save_dir[i])
                print(f'创建{dirs[i]}文件夹成功！')
            image_list, label_list = getdata.get_file(src)
            convert_to_tfrecord(image_list, label_list, save_dir[i], dirs[i])
    run()
