#! /user/bin/python3
# -*- coding:utf-8 -*-
# @Author  : Paul C G LUO
# @Email   : 673951437@qq.com
# @DateTime: 2021/7/13 14:11
from trainDataProcess.dataLoading import parse_annotation
import tensorflow as tf2


def preprocess(img, img_boxes):
    """

    :param img: string, it is the path of image.
    :param img_boxes: the shape is [40, 5]
    :return:
    """
    x = tf2.io.read_file(img)
    x = tf2.image.decode_png(x, channels=3)
    x = tf2.image.convert_image_dtype(x, tf2.float32)

    return x, img_boxes


def get_dataset(img_dir, ann_dir, labels, batch_size):
    """

    :param img_dir:
    :param ann_dir:
    :param batch_size:
    :return: tensorflow dataset
    """
    imgs, boxes = parse_annotation(img_dir, ann_dir, labels)
    db = tf2.data.Dataset.from_tensor_slices((imgs, boxes))
    db = db.shuffle(1000).map(preprocess).batch(batch_size).repeat()
    print(len(imgs))
    return db


if __name__ == '__main__':
    labels = ['sugarbeet', 'weed']
    a = get_dataset('D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\image',
                                   'D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\annotation',
                                   labels, 4)

    print(a)