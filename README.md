# DataAugYolo

## 简介

本项目的功能是对YOLO格式的数据实现数据增强，使用本项目代码可以较快的完成**扩充数据集**的工作，会对图像和标注同时处理。但其也存在一定的问题，例如无法保证处理后的数据一定能跑出更好的效果，处理结束后**可能会出现一些损坏的图片**（原图过大时）。

但本项目处理后出现的坏图，在YOLOv5中会被识别出来并不做训练，其实对训练影响不大，追求完美的同学可以考虑不使用resize系列的函数，手动剔除坏图，使用其他数据增强项目完成工作等。

## 使用

1. download本项目到本地(建议下载到有图形界面的操作系统中)

2. 打开DataAugOnDetectin.py

   修改 image_path， label_path， save_path 三个参数

   ```
   image_path = ""   # 图片的路径 
   label_path = ""	  # 标签文件的路径
   save_path = ""    # 数据增强的结果保存位置路径
   ```

3. 运行，使用pycharm或spyder等软件运行DataAugOnDetectin.py

   也可以用命令行运行 

   ```
   > cd DataAugYolo
   > python DataAugOnDetection.py
   ```

注：本项目自带了一个小小的数据集，可以直接运行来看一下效果，如果想看boxes是否出错，可以使用plot_pics函数来查看

## 函数功能简介

**DataAugmentationOnDetection类**中的函数

| 序号 |        函数名         | 功能                                                         |
| :--: | :-------------------: | :----------------------------------------------------------- |
|  1   |   resize_keep_ratio   | 将图像最长边缩放到size尺寸，同时保持长宽比                   |
|  2   | resizeDown_keep_ratio | 将图像最长边缩减到size尺寸，保持长宽比，小于size尺寸的图片不进行处理 |
|  3   |        resize         | 将图像长和宽缩放到指定值size                                 |
|  4   |  random_flip_horizon  | 将图像水平翻转（镜像）                                       |
|  5   | random_flip_vertical  | 将图像竖直翻转（倒立）                                       |
|  6   |      center_crop      | 中心裁剪，如果设置了Target_size参数，函数会将裁剪后的图像缩放到（target_size, Target_size)的尺寸 |
|  7   |     random_bright     | 改变图像亮度                                                 |
|  8   |    random_contrast    | 改变对比度                                                   |
|  9   |   random_saturation   | 改变对比度                                                   |
|  11  |   add_gasuss_noise    | 添加高斯噪声                                                 |
|  12  |    add_salt_noise     | 添加盐噪声                                                   |
|  13  |   add_pepper_noise    | 添加胡椒噪音                                                 |

**runAugumentation函数：**

运行数据增强的函数，集成实现了DataAugmentationOnDetection中的大部分函数，直接运行这个函数是可以正确完成数据增强工作的，使用者可以根据注释来增加或者删除一些增强的操作。

其他函数均是一些不重要的函数，且代码中已经注释其功能。

## 其他

**项目的参考：**   https://github.com/REN-HT/Data-Augmentation/blob/main/data_augmentation.py

根据上述项目修改而来，上述项目可以完成VOC格式数据的数据增强，本项目修改部分内容后可以完成YOLO格式数的数据增强，同时修改了原项目中一些小错误，增加了文件流等

**项目的主要用途：**

做这个项目的时候是因为我在训练YOLOv5时遇到了某类样本不够的问题，自己手标数据感觉很累，因此采用数据增强的方式扩充了数据量，减少了标注成本。

**项目的正确性：**

虽然我每个函数都进行了测试，目前看来也没有坐标计算错误，其实正确性也是难以保证，例如图像文件过大的时候python库可能会无法处理这些图像，如果发现不合理的地方（boxes坐标计算问题），可以反馈给我，如果这个项目发生了错误，给您带来了困扰，我在此表示抱歉。