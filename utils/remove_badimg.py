'''
Author: Ken Kaneki
Date: 2021-12-04 23:28:52
LastEditTime: 2021-12-05 17:10:04
Description: README
'''
import glob
import os
import warnings

from PIL import Image

warnings.filterwarnings("error", category=UserWarning)



def remove_badimg(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    for f in cate:
        #print(f)
        for im in glob.glob(f + '/*.*'):
            #print(im)
            try:
                img = Image.open(im)
            except Exception:
                print('corrupt img', im)
                print(img)
                os.remove(im)
img_train_path = '../datasets/data3/TRAIN/'
img_test_path = '../datasets/data3/TEST/'
img_test_PART_path = '../datasets/data3/TEST_PART/'
remove_badimg(img_train_path)
remove_badimg(img_test_path)
remove_badimg(img_test_PART_path)
