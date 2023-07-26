# -*- coding: utf-8 -*-
"""
Created on 2023-04-01 9:08

@author: Fan yi ming

Func: 对于目标检测的数据增强[YOLO]（特点是数据增强后标签也要更改）
review：常用的数据增强方式；
        1.翻转：左右和上下翻转，随机翻转
        2.随机裁剪，图像缩放
        3.改变色调
        4.添加噪声

注意： boxes的标签和坐标一个是int，一个是float，存放的时候要注意处理方式。

参考：https://github.com/REN-HT/Data-Augmentation/blob/main/data_augmentation.py
"""
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
random.seed(0)


class DataAugmentationOnDetection:
    def __init__(self):
        super(DataAugmentationOnDetection, self).__init__()

    # 以下的几个参数类型中，image的类型全部如下类型
    # 参数类型： image：Image.open(path)
    def resize_keep_ratio(self, image, boxes, target_size):
        """
            参数类型： image：Image.open(path)， boxes:Tensor， target_size:int
            功能：将图像缩放到size尺寸，调整相应的boxes,同时保持长宽比（最长的边是target size
        """
        old_size = image.size[0:2]  # 原始图像大小
        # 取最小的缩放比例
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
        new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
        # boxes 不用变化，因为是等比例变化
        return image.resize(new_size, Image.BILINEAR), boxes

    def resizeDown_keep_ratio(self, image, boxes, target_size):
        """ 与上面的函数功能类似，但它只降低图片的尺寸，不会扩大图片尺寸"""
        old_size = image.size[0:2]  # 原始图像大小
        # 取最小的缩放比例
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
        ratio = min(ratio, 1)
        new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小

        # boxes 不用变化，因为是等比例变化
        return image.resize(new_size, Image.BILINEAR), boxes

    def resize(self, img, boxes, size):
        # ---------------------------------------------------------
        # 类型为 img=Image.open(path)，boxes:Tensor，size:int
        # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
        # ---------------------------------------------------------
        return img.resize((size, size), Image.BILINEAR), boxes

    def random_flip_horizon(self, img, boxes, h_rate=1):
        # -------------------------------------
        # 随机水平翻转
        # -------------------------------------
        if np.random.random() < h_rate:
            transform = transforms.RandomHorizontalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                x = 1 - boxes[:, 1]
                boxes[:, 1] = x
        return img, boxes

    def random_flip_vertical(self, img, boxes, v_rate=1):
        # 随机垂直翻转
        if np.random.random() < v_rate:
            transform = transforms.RandomVerticalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                y = 1 - boxes[:, 2]
                boxes[:, 2] = y
        return img, boxes

    def center_crop(self, img, boxes, target_size=None):
        # -------------------------------------
        # 中心裁剪 ，裁剪成 (size, size) 的正方形, 仅限图形，w,h
        # 这里用比例是很难算的，转成x1,y1, x2, y2格式来计算
        # -------------------------------------
        w, h = img.size
        size = min(w, h)
        if len(boxes) > 0:
            # 转换到xyxy格式
            label = boxes[:, 0].reshape([-1, 1])
            x_, y_, w_, h_ = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            x1 = (w * x_ - 0.5 * w * w_).reshape([-1, 1])
            y1 = (h * y_ - 0.5 * h * h_).reshape([-1, 1])
            x2 = (w * x_ + 0.5 * w * w_).reshape([-1, 1])
            y2 = (h * y_ + 0.5 * h * h_).reshape([-1, 1])
            boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)
            # 边框转换
            if w > h:
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - (w - h) / 2
            else:
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - (h - w) / 2
            in_boundary = [i for i in range(boxes_xyxy.shape[0])]
            for i in range(boxes_xyxy.shape[0]):
                # 判断x是否超出界限
                if (boxes_xyxy[i, 0] < 0 and boxes_xyxy[i, 2] < 0) or (boxes_xyxy[i, 0] > size and boxes_xyxy[i, 2] > size):
                    in_boundary.remove(i)
                # 判断y是否超出界限
                elif (boxes_xyxy[i, 1] < 0 and boxes_xyxy[i, 3] < 0) or (boxes_xyxy[i, 1] > size and boxes_xyxy[i, 3] > size):
                    in_boundary.append(i)
            boxes_xyxy = boxes_xyxy[in_boundary]
            boxes = boxes_xyxy.clamp(min=0, max=size).reshape([-1, 4])  # 压缩到固定范围
            label = label[in_boundary]
            # 转换到YOLO格式
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            xc = ((x1 + x2) / (2 * size)).reshape([-1, 1])
            yc = ((y1 + y2) / (2 * size)).reshape([-1, 1])
            wc = ((x2 - x1) / size).reshape([-1, 1])
            hc = ((y2 - y1) / size).reshape([-1, 1])
            boxes = torch.cat([xc, yc, wc, hc], dim=1)
        # 图像转换
        transform = transforms.CenterCrop(size)
        img = transform(img)
        if target_size:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        if len(boxes) > 0:
            return img, torch.cat([label.reshape([-1, 1]), boxes], dim=1)
        else:
            return img, boxes

    # ------------------------------------------------------
    # 以下img皆为Tensor类型
    # ------------------------------------------------------

    def random_bright(self, img, u=120, p=1):
        # -------------------------------------
        # 随机亮度变换
        # -------------------------------------
        if np.random.random() < p:
            alpha=np.random.uniform(-u, u)/255
            img += alpha
            img=img.clamp(min=0.0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5, p=1):
        # -------------------------------------
        # 随机增强对比度
        # -------------------------------------
        if np.random.random() < p:
            alpha=np.random.uniform(lower, upper)
            img*=alpha
            img=img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img,lower=0.5, upper=1.5, p=1):
        # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
        if np.random.random() < p:
            alpha=np.random.uniform(lower, upper)
            img[1]=img[1]*alpha
            img[1]=img[1].clamp(min=0,max=1.0)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        noise=torch.normal(mean,std,img.shape)
        img+=noise
        img=img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()/5 + 0.7
        img[noise[:,:,:]>alpha]=1.0
        return img

    def add_pepper_noise(self, img):
        noise=torch.rand(img.shape)
        alpha=np.random.random()/5 + 0.7
        img[noise[:, :, :]>alpha]=0
        return img


def plot_pics(img, boxes):
    # 显示图像和候选框，img是Image.Open()类型, boxes是Tensor类型
    plt.imshow(img)
    label_colors = [(213, 110, 89)]
    w, h = img.size
    for i in range(boxes.shape[0]):
        box = boxes[i, 1:]
        xc, yc, wc, hc = box
        x = w * xc - 0.5 * w * wc
        y = h * yc - 0.5 * h * hc
        box_w, box_h = w * wc, h * hc
        plt.gca().add_patch(plt.Rectangle(xy=(x, y), width=box_w, height=box_h,
                                          edgecolor=[c / 255 for c in label_colors[0]],
                                          fill=False, linewidth=2))
    plt.show()

def get_image_list(image_path):
    # 根据图片文件，查找所有图片并返回列表
    files_list = []
    for root, sub_dirs, files in os.walk(image_path):
        for special_file in files:
            special_file = special_file[0: len(special_file)]
            files_list.append(special_file)
    return files_list

def get_label_file(label_path, image_name):
    # 根据图片信息，查找对应的label
    fname = os.path.join(label_path, image_name[0: len(image_name)-4]+".txt")
    data2 = []
    if not os.path.exists(fname):
        return data2
    if os.path.getsize(fname) == 0:
        return data2
    else:
        with open(fname, 'r', encoding='utf-8') as infile:
            # 读取并转换标签
            for line in infile:
                data_line = line.strip("\n").split()
                data2.append([float(i) for i in data_line])
    return data2


def save_Yolo(img, boxes, save_path, prefix, image_name):
    # img: 需要时Image类型的数据， prefix 前缀
    # 将结果保存到save path指示的路径中
    if not os.path.exists(save_path) or \
            not os.path.exists(os.path.join(save_path, "images")):
        os.makedirs(os.path.join(save_path, "images"))
        os.makedirs(os.path.join(save_path, "labels"))
    try:
        img.save(os.path.join(save_path, "images", prefix + image_name))
        with open(os.path.join(save_path, "labels", prefix + image_name[0:len(image_name)-4] + ".txt"), 'w', encoding="utf-8") as f:
            if len(boxes) > 0:  # 判断是否为空
                # 写入新的label到文件中
                for data in boxes:
                    str_in = ""
                    for i, a in enumerate(data):
                        if i == 0:
                            str_in += str(int(a))
                        else:
                            str_in += " " + str(float(a))
                    f.write(str_in + '\n')
    except:
        print("ERROR: ", image_name, " is bad.")


def runAugumentation(image_path, label_path, save_path):
    image_list = get_image_list(image_path)
    for image_name in image_list:
        print("dealing: " + image_name)
        img = Image.open(os.path.join(image_path, image_name))
        boxes = get_label_file(label_path, image_name)
        boxes = torch.tensor(boxes)
        # 下面是执行的数据增强功能，可自行选择
        # Image类型的参数
        DAD = DataAugmentationOnDetection()

        """ 尺寸变换   """
        # 缩小尺寸
        # t_img, t_boxes = DAD.resizeDown_keep_ratio(img, boxes, 1024)
        # save_Yolo(t_img, boxes, save_path, prefix="rs_", image_name=image_name)
        # 水平旋转
        t_img, t_boxes = DAD.random_flip_horizon(img, boxes.clone())
        save_Yolo(t_img, t_boxes, save_path, prefix="fh_", image_name=image_name)
        # 竖直旋转
        t_img, t_boxes = DAD.random_flip_vertical(img, boxes.clone())
        save_Yolo(t_img, t_boxes, save_path, prefix="fv_", image_name=image_name)
        # center_crop
        t_img, t_boxes = DAD.center_crop(img, boxes.clone(), 1024)
        save_Yolo(t_img, t_boxes, save_path, prefix="cc_", image_name=image_name)

        """ 图像变换，用tensor类型"""
        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()
        img = to_tensor(img)

        # random_bright
        t_img, t_boxes = DAD.random_bright(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rb_", image_name=image_name)
        # random_contrast 对比度变化
        t_img, t_boxes = DAD.random_contrast(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rc_", image_name=image_name)
        # random_saturation 饱和度变化
        t_img, t_boxes = DAD.random_saturation(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rs_", image_name=image_name)
        # 高斯噪声
        t_img, t_boxes = DAD.add_gasuss_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="gn_", image_name=image_name)
        # add_salt_noise
        t_img, t_boxes = DAD.add_salt_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="sn_", image_name=image_name)
        # add_pepper_noise
        t_img, t_boxes = DAD.add_pepper_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="pn_", image_name=image_name)

        print("end:     " + image_name)


if __name__ == '__main__':
    # 图像和标签文件夹
    image_path = "./TestYoloSet/images"
    label_path = "./TestYoloSet/labels"
    save_path = "./TestYoloSet/Augumentation"    # 结果保存位置路径，可以是一个不存在的文件夹
    # 运行
    runAugumentation(image_path, label_path, save_path)
