import warnings
import os
import glob
from PIL import Image
warnings.filterwarnings("error", category=UserWarning)

path = 'imagpath'

cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
for f in cate:
    # print(f)
    for im in glob.glob(f + '/*.jpg'):
        # print(im)
        try:
            img = Image.open(im)
        except:
            print('corrupt img', im)
            print(img)
            os.remove(im)
