#! /user/bin/python3
# -*- coding:utf-8 -*-
# @Author  : Paul C G LUO
# @Email   : 673951437@qq.com
# @DateTime: 2021/7/12 20:52

import os

import tensorflow as tf2
import numpy as np
import xml.etree.ElementTree as ET


def parse_annotation(img_dir, ann_dir, labels):
    """
    parse xml files
    :param img_dir: images path
    :param ann_dir: annotation xml files path
    :param labels: ('sugarbeet', 'weed')
    :return:
    """
    imgs_info = []
    max_boxes = 0

    # for each annotation xml files

    for ann in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann))
        img_info = dict()
        img_info['object'] = []

        boxes_counter = 0
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_info['filename'] = os.path.join(img_dir, elem.text)

            if 'width' in elem.tag:
                img_info['width'] = int(elem.text)
                assert img_info['width'] == 512

            if 'height' in elem.tag:
                img_info['height'] = int(elem.text)
                assert img_info['height'] == 512

            if 'object' in elem.tag or 'part' in elem.tag:
                object_info = [0.0, 0.0, 0.0, 0.0, 0.0]  # x1-y1-x2-y2-label
                boxes_counter += 1

                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = labels.index(attr.text) + 1
                        object_info[4] = label

                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)  # filename w/h/box_info
            if boxes_counter > max_boxes:
                max_boxes = boxes_counter

        imgs_info.append(img_info)


    # the maximum boxes number is max_boxes
    # [b, 40, 5]
    boxes = np.zeros([len(imgs_info), max_boxes, 5])
    imgs = []

    for idx, single_img in enumerate(imgs_info):
        print('idx:', idx)
        # [N, 5]
        img_boxes = np.array(single_img['object'])
        # overwrite the N boxes information

        boxes[idx, :img_boxes.shape[0]] = img_boxes
        imgs.append(single_img['filename'])

        print(single_img['filename'], boxes[idx, :5], len(boxes[idx]))

    return imgs, boxes


if __name__ == '__main__':
    labels = ['sugarbeet', 'weed']
    imgs, boxes = parse_annotation('D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\image',
                                   'D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\annotation',
                                   labels)

    print(boxes.shape)
