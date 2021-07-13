#! /user/bin/python3
# -*- coding:utf-8 -*-
# @Author  : Paul C G LUO
# @Email   : 673951437@qq.com
# @DateTime: 2021/7/13 17:13

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import tensorflow as tf2

from trainDataProcess.generateBatchData import get_dataset
from trainDataProcess.imageVisual import db_visualize


def augmentation_generator(yolo_dataset):
    """
    augmented batch generator from a yolo dataset
    :param yolo_dataset:
    :return:
        augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    """
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                     y1=bb[1],
                                     x2=bb[2],
                                     y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] + bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(512, 512)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)),  # change brightness
            # iaa.ContrastNormalization((0.5, 1.5)),
            # iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
        ])
        # seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i, j, 0] = bb.x1
                boxes[i, j, 1] = bb.y1
                boxes[i, j, 2] = bb.x2
                boxes[i, j, 3] = bb.y2
        # conversion numpy->tensor
        batch = (tf2.convert_to_tensor(img_aug), tf2.convert_to_tensor(boxes))
        # batch = (img_aug, boxes)
        yield batch


if __name__ == '__main__':
    labels = ['sugarbeet', 'weed']
    train_db = get_dataset('D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\image',
                           'D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\annotation',
                           labels, 4)

    aug_train_db = augmentation_generator(train_db)
    db_visualize(aug_train_db)
