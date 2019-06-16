#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang


import cv2
import os

import os


def resize(src):
    """
    按照src的目录结构创建剪裁成227*227的图片
    :param src: 需要裁剪图片的根目录
    :return:
    """
    succes = 0
    fail = 0
    fail_file = []
    for root, dirs, files in os.walk(src):
        print('开始文件写入……')
        for file in files:
            filepath = os.path.join(root, file)
            filepath_list = filepath.split('\\')
            # print(filepath)
            # print(filepath_list)
            # print(file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                cwd = os.getcwd()
                new_img_dir = os.path.join(cwd, 'resized_images2')
                # new_img_dir_test = os.path.join(new_img_dir, filepath_list[-3]) 直接在这个地址创建文件夹报错
                # print('test:' + new_img_dir_test)
                if not os.path.exists(new_img_dir):
                    os.mkdir(new_img_dir)
                    # print('success')
                new_img_path = new_img_dir + os.sep + filepath_list[-3]
                # print(new_img_path == new_img_dir_test)
                if not os.path.exists(new_img_path):
                    os.mkdir(new_img_path)
                class_name = new_img_path + os.sep + filepath_list[-2]
                if not os.path.exists(class_name):
                    print('{}文件夹不存在，已创建'.format(class_name))
                    os.mkdir(class_name)
                path = os.path.join(class_name, file)
                if not os.path.exists(path):
                    cv2.imwrite(path, resized)
                    succes += 1
                    # print('写入文件{}成功'.format(path))
                    # pass
            except:
                fail += 1
                path += '\\n'
                fail_file.append(path)
                print(filepath + '文件出错')
                if (succes + fail) % 500 == 0:
                    print('已处理{}张文件，成功{}，失败{}，失败文件请查看fail.txt'.format(succes+fail, succes, fail))
            finally:
                f = open('fail.txt', 'w')
                f.write(fail_file)
    print('总共成功写入{}张，失败{}'.format(succes, fail))



if __name__ == '__main__':
    path = r'D:\DataSet\kaggle\catdog'
    resize(path)