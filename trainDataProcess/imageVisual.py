#! /user/bin/python3
# -*- coding:utf-8 -*-
# @Author  : Paul C G LUO
# @Email   : 673951437@qq.com
# @DateTime: 2021/7/13 14:49

from matplotlib import pyplot as plt
from matplotlib import patches

from trainDataProcess.generateBatchData import get_dataset


def db_visualize(db):
    """
    imgs:　　　　　[b, 512, 512, 3]
    imgs_boxes:　[b, 40, 5]

    :param db:
    :return:
    """
    imgs, imgs_boxes = next(iter(db))
    img, img_boxes = imgs[0], imgs_boxes[0]

    f, ax1 = plt.subplots(1, figsize=(10, 10))

    ax1.imshow(img)
    for x1, y1, x2, y2, l in img_boxes:
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        w = x2 - x1
        h = y2 - y1

        if l == 1:
            color = (0, 1, 0)
        elif l == 2:
            color = (1, 0, 0)  # RGB
        else:
            break

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    labels = ['sugarbeet', 'weed']
    db = get_dataset('D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\image',
                     'D:\\Tensorflow_version2.0\\［更新］目标检测\\yolov2-tf2\\yolov2-tf2\\data\\train\\annotation',
                     labels, 4)

    db_visualize(db)
