#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author:Dr.Shang
import numpy as np


def onehot(labels):
    num_sample = len(labels)
    num_class = max(labels) + 1
    onehot_labels = np.zeros((num_sample, num_class))
    onehot_labels[np.arange(num_sample), labels] = 1
    return onehot_labels