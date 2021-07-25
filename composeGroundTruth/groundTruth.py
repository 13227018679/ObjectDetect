#! /user/bin/python3
# -*- coding:utf-8 -*-
# @Author  : Paul C G LUO
# @Email   : 673951437@qq.com
# @DateTime: 2021/7/14 14:29

import numpy as np
import tensorflow as tf2
from matplotlib import pyplot as plt

from trainDataProcess.dataAugmentation import augmentation_generator
from trainDataProcess.generateBatchData import get_dataset

IMAGESIZE = 512
GRIDSIZE = 16
ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS_NUM = len(ANCHORS) // 2


def process_true_boxes(gt_boxes, anchors):
    # 512 // 16 = 32
    scale = IMAGESIZE // GRIDSIZE
    # [5, 2]
    anchors = np.array(anchors).reshape((ANCHORS_NUM, 2))

    detector_mask = np.zeros([GRIDSIZE, GRIDSIZE, 5, 1])

    matching_gt_box = np.zeros([GRIDSIZE, GRIDSIZE, 5, 5])

    gt_boxes_grid = np.zeros(gt_boxes.shape)

    # DB: tensor => numpy
    gt_boxes = gt_boxes.numpy()

    for i, box in enumerate(gt_boxes):  # [40, 5]
        # box: [5], x1-y1-x2-y2-l
        # 0~512 -> 0~16
        x = ((box[0] + box[2]) / 2) / scale
        y = ((box[1] + box[3]) / 2) / scale
        w = (box[2] - box[0]) / scale
        h = (box[3] - box[1]) / scale

        gt_boxes_grid[i] = np.array([x, y, w, h, box[4]])

        if w * h > 0:
            best_anchor = 0
            best_iou = 0
            for j in range(5):
                interct = np.minimum(w, anchors[j, 0]) * np.minimum(h, anchors[j, 1])
                union = w * h + (anchors[j, 0] * anchors[j, 1]) - interct
                iou = interct / union

                if iou > best_iou:
                    best_anchor = j
                    best_iou = iou

            # found the best anchors
            if best_iou > 0:
                x_coord = np.floor(x).astype(np.int32)
                y_coord = np.floor(y).astype(np.int32)
                # [b,h,w,5,1]
                detector_mask[y_coord, x_coord, best_anchor] = 1
                # [b,h,w,5,x-y-w-h-l]
                matching_gt_box[y_coord, x_coord, best_anchor] = \
                    np.array([x, y, w, h, box[4]])

    return matching_gt_box, detector_mask, gt_boxes_grid

def ground_truth_generator(db):

    for imgs, imgs_boxes in db:
        # imgs: [b, 512, 512, 3]
        # imgs_boxes: [b, 40, 5]

        batch_matching_gt_box = []
        batch_detector_mask = []
        batch_gt_boxes_grid= []

        b = imgs.shape[0]
        for i in range(b):
            matching_gt_box, detector_mask, gt_boxes_grid = process_true_boxes(imgs_boxes[i], ANCHORS)
            batch_matching_gt_box.append(matching_gt_box)
            batch_detector_mask.append(detector_mask)
            batch_gt_boxes_grid.append(gt_boxes_grid)

        # [b, 16,16,5,1]
        detector_mask = tf2.cast(np.array(batch_detector_mask), dtype=tf2.float32)
        # [b,16,16,5,5] x-y-w-h-l
        matching_gt_box = tf2.cast(np.array(batch_matching_gt_box), dtype=tf2.float32)
        # [b,40,5] x-y-w-h-l
        gt_boxes_grid = tf2.cast(np.array(batch_gt_boxes_grid), dtype=tf2.float32)

        # [b,16,16,5]
        matching_classes = tf2.cast(matching_gt_box[..., 4], dtype=tf2.int32)
        # [b,16,16,5,3]
        matching_classes_oh = tf2.one_hot(matching_classes, depth=3)
        # x-y-w-h-conf-l1-l2
        # [b,16,16,5,2]
        matching_classes_oh = tf2.cast(matching_classes_oh[..., 1:], dtype=tf2.float32)
        # [b,512,512,3]
        # [b,16,16,5,1]
        # [b,16,16,5,5]
        # [b,16,16,5,2]
        # [b,40,5]
        yield imgs, detector_mask, matching_gt_box, matching_classes_oh, gt_boxes_grid


if __name__ == '__main__':
    # %%
    # 2.3 visualize object mask
    # train_db -> aug_train_db -> train_gen
    labels = ['sugarbeet', 'weed']
    train_db = get_dataset('D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\image',
                           'D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\annotation',
                           labels, 4)

    aug_train_db = augmentation_generator(train_db)
    train_gen = ground_truth_generator(aug_train_db)

    img, detector_mask, matching_gt_box, matching_classes_oh, gt_boxes_grid = \
        next(train_gen)
    img, detector_mask, matching_gt_box, matching_classes_oh, gt_boxes_grid = \
        img[0], detector_mask[0], matching_gt_box[0], matching_classes_oh[0], gt_boxes_grid[0]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 10))
    ax1.imshow(img)
    # [16,16,5,1] => [16,16,1]
    mask = tf2.reduce_sum(detector_mask, axis=2)
    ax2.matshow(mask[..., 0])  # [16,16]
    plt.show()
