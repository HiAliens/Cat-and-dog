#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import os
import numpy as np


def get_file(file_dir):
    '''

    根据指定的训练或验证或测试数据集的路径获取图片集,目录结构应为 file——dir下包含cat和dog两个文件夹
    :param file_dir: 训练数据集文件夹
    :param is_tfr:  以何种方法得到image和label
    :return: image和label
    '''
    images = []
    floders = []
    for root, sub_folders, files in os.walk(file_dir):
        for file in files:
            images.append(os.path.join(root, file))
        for floder in sub_folders:
            floders.append(os.path.join(root, floder))

    labels = []
    for one_floder in floders:
        num_img = len(os.listdir(one_floder)) # 统计one_floder下包含多少个文件
        label = one_floder.split('\\')[-1]
        # print(label)
        if label == 'cats':
            labels = np.append(labels, num_img * [0]) # 生成一个2维列表
        else:
            labels = np.append(labels, num_img * [1])
    # print(len(labels))

    # shuffle
    temp = []
    temp = np.array([images, labels])
    # print(temp)
    temp = temp.transpose()
    # print(temp)
    '''
    [['D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1500.jpg'
  'D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1501.jpg'
  'D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1502.jpg' ...
  'D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1997.jpg'
  'D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1998.jpg'
  'D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1999.jpg']
 ['0.0' '0.0' '0.0' ... '1.0' '1.0' '1.0']]
[['D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1500.jpg' '0.0']
 ['D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1501.jpg' '0.0']
 ['D:\\DataSet\\kaggle\\small_samples\\test\\cats\\cat.1502.jpg' '0.0']
 ...
 ['D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1997.jpg' '1.0']
 ['D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1998.jpg' '1.0']
 ['D:\\DataSet\\kaggle\\small_samples\\test\\dogs\\dog.1999.jpg' '1.0']]
    '''
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


if __name__ == '__main__':
    image, label = get_file(r'D:\coding\python\coding-pycharm\opencv+tensorflow\CAT-VS-DOG\resized_images2\test')
    # print(image)
    # print(label)