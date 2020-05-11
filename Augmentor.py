from imgaug import augmenters as iaa
import os
import cv2
from PIL import Image
import numpy as np
path = r'D:\E\document\datas\DoubleEyelid\train'
path1 = r'D:\E\document\datas\DoubleEyelid\a1'

for i in range(1):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        iaa.Sharpen(0.5),
        # img = open(os.path.join(path, "1.jpg"))
        # print(img)
        # iaa.GaussianBlur(0),
        # iaa.AverageBlur(),
        # iaa.Affine(shear=25, mode=["edge"])
    ])
    # imglist = []
    for imgs_path in os.listdir(path):
        # print(imgs_path + '1')
        for img_path in os.listdir(os.path.join(path, imgs_path)):
            # print(img_path)
            imglist = []
            img = cv2.imread(os.path.join(path, imgs_path, img_path))
            # print(img)
            imglist.append(img)
            # print(imglist)
            try:
                images_aug = seq.augment_images(imglist)
            except AttributeError as e:
                print("报错信息：", e, "出错的图片为：", img_path)
            img_path_ = img_path.split(".")[0] + '{}'.format(i)
            cv2.imwrite(os.path.join(path1, imgs_path, "{}.png".format(img_path_)), images_aug[0])
for i in range(1, 2):
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        # iaa.Sharpen(0.5),
        # img = open(os.path.join(path, "1.jpg"))
        # print(img)
        iaa.GaussianBlur(0),
        # iaa.AverageBlur(),
        iaa.Affine(shear=25, mode=["edge"])
    ])
    # imglist = []
    for imgs_path in os.listdir(path):
        # print(imgs_path + '1')
        for img_path in os.listdir(os.path.join(path, imgs_path)):
            # print(img_path)
            imglist = []
            img = cv2.imread(os.path.join(path, imgs_path, img_path))
            # print(img)
            imglist.append(img)
            # print(imglist)
            try:
                images_aug = seq.augment_images(imglist)
            except AttributeError as e:
                print("报错信息：", e, "出错的图片为：", img_path)
            img_path_ = img_path.split(".")[0] + '{}'.format(i)
            cv2.imwrite(os.path.join(path1, imgs_path, "{}.png".format(img_path_)), images_aug[0])

for i in range(2, 3):
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # iaa.Sharpen(0.5),
        # img = open(os.path.join(path, "1.jpg"))
        # print(img)
        # iaa.GaussianBlur(0),
        iaa.AverageBlur(),
        # iaa.Affine(shear=25, mode=["edge"])
    ], random_order=True)
    # imglist = []
    for imgs_path in os.listdir(path):
        # print(imgs_path + '1')
        for img_path in os.listdir(os.path.join(path, imgs_path)):
            # print(img_path)
            imglist = []
            img = cv2.imread(os.path.join(path, imgs_path, img_path))
            # print(img)
            imglist.append(img)
            # print(imglist)
            try:
                images_aug = seq.augment_images(imglist)
            except AttributeError as e:
                print("报错信息：", e, "出错的图片为：", img_path)
            img_path_ = img_path.split(".")[0] + '{}'.format(i)
            cv2.imwrite(os.path.join(path1, imgs_path, "{}.png".format(img_path_)), images_aug[0])
# vertically flip each input image with 90% probability
#     images_aug = iaa.Flipud(0.9)(images=images)

# blur 50% of all images using a gaussian kernel with a sigma of 3.0
#     images_aug = iaa.Sometimes(0.5, iaa.GaussianBlur(3.0))(images=images)
#     images_aug.save(os.path.join(path1, "{}".format(i)))
