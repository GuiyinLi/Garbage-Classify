'''
Author: Ken Kaneki
Date: 2021-12-04 21:38:52
LastEditTime: 2021-12-06 09:21:33
Description: 从"img_path"文件夹随机选取70%的图片移动到"copy_to_path"文件夹中
'''
import os
import random
import shutil

datasets_test_path="E:/VSCode/Python/PC-Garbage-classify/datasets/data3/TEST/"
datasets_train_path="E:/VSCode/Python/PC-Garbage-classify/datasets/data3/TRAIN/"
# 分组
def image_group(img_root_path,copy_root_path):
    for root, dirs, files in os.walk(img_root_path, topdown=True):
        for dirname in dirs:
            IMG_ROOT_DIR = os.path.abspath(img_root_path)
            COPY_ROOT_DIR = os.path.abspath(copy_root_path)

            img_path = os.path.join(IMG_ROOT_DIR, dirname)
            copy_to_path = os.path.join(COPY_ROOT_DIR, dirname)

            if not os.path.exists(copy_to_path):
                os.makedirs(copy_to_path)

            imglist = os.listdir(img_path)
            random_imglist = random.sample(imglist, int(0.7*len(imglist)))
            for img in random_imglist:
                # 图片复制到另一个文件夹
                shutil.copy(os.path.join(img_path, img), os.path.join(copy_to_path, img))
                os.remove(os.path.join(img_path, img))#并删除原有文件

# 计数
def image_count(root_path):
    num=0
    for root, dirs, files in os.walk(root_path, topdown=True):
        for dirname in dirs:
            img_path = os.path.join(root_path, dirname)
            imglist = os.listdir(img_path)
            num=int(len(imglist))+num
    return num

if __name__ == '__main__':
    #image_group(datasets_test_path,datasets_train_path)
    #print(image_count(datasets_test_path))
    print('test_image_num: %d' % image_count(datasets_test_path))
    print('test_image_num: %d' % image_count(datasets_train_path))
