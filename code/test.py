﻿'''
Author: Ken Kaneki
Date: 2021-12-04 18:13:42
LastEditTime: 2021-12-06 08:37:03
Description: README
'''
import csv
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_data1 = np.array(['cardboard', 'glass','metal','paper','plastic','trash'])
model_data1="trash_data1_AlexNet"
test_path_data1='../datasets/data1/TEST/'

label_data2 = np.array(['O', 'R'])
model_data2="trash_data2_AlexNet"
test_path_data2='../datasets/data2/TEST/'

label_data3_all = ['其他垃圾_PE塑料袋', '其他垃圾_U型回形针', '其他垃圾_一次性杯子', '其他垃圾_一次性棉签',
           '其他垃圾_串串竹签', '其他垃圾_便利贴', '其他垃圾_创可贴', '其他垃圾_厨房手套', '其他垃圾_口罩',
           '其他垃圾_唱片', '其他垃圾_图钉', '其他垃圾_大龙虾头', '其他垃圾_奶茶杯', '其他垃圾_干果壳',
           '其他垃圾_干燥剂', '其他垃圾_打泡网', '其他垃圾_打火机', '其他垃圾_放大镜', '其他垃圾_毛巾',
           '其他垃圾_涂改带', '其他垃圾_湿纸巾', '其他垃圾_烟蒂', '其他垃圾_牙刷', '其他垃圾_百洁布',
           '其他垃圾_眼镜', '其他垃圾_票据', '其他垃圾_空调滤芯', '其他垃圾_笔及笔芯', '其他垃圾_纸巾',
           '其他垃圾_胶带', '其他垃圾_胶水废包装', '其他垃圾_苍蝇拍', '其他垃圾_茶壶碎片', '其他垃圾_餐盒',
           '其他垃圾_验孕棒', '其他垃圾_鸡毛掸', '厨余垃圾_八宝粥', '厨余垃圾_冰糖葫芦', '厨余垃圾_咖啡渣',
           '厨余垃圾_哈密瓜', '厨余垃圾_圣女果', '厨余垃圾_巴旦木', '厨余垃圾_开心果', '厨余垃圾_普通面包',
           '厨余垃圾_板栗', '厨余垃圾_果冻', '厨余垃圾_核桃', '厨余垃圾_梨', '厨余垃圾_橙子', '厨余垃圾_残渣剩饭',
           '厨余垃圾_汉堡', '厨余垃圾_火龙果', '厨余垃圾_炸鸡', '厨余垃圾_烤鸡烤鸭', '厨余垃圾_牛肉干', '厨余垃圾_瓜子',
           '厨余垃圾_甘蔗', '厨余垃圾_生肉', '厨余垃圾_番茄', '厨余垃圾_白菜', '厨余垃圾_白萝卜', '厨余垃圾_粉条',
           '厨余垃圾_糕点', '厨余垃圾_红豆', '厨余垃圾_肠(火腿)', '厨余垃圾_胡萝卜', '厨余垃圾_花生皮', '厨余垃圾_苹果',
           '厨余垃圾_茶叶', '厨余垃圾_草莓', '厨余垃圾_荷包蛋', '厨余垃圾_菠萝', '厨余垃圾_菠萝包', '厨余垃圾_菠萝蜜',
           '厨余垃圾_蒜', '厨余垃圾_薯条', '厨余垃圾_蘑菇', '厨余垃圾_蚕豆', '厨余垃圾_蛋', '厨余垃圾_蛋挞', '厨余垃圾_西瓜皮',
           '厨余垃圾_贝果', '厨余垃圾_辣椒', '厨余垃圾_陈皮', '厨余垃圾_青菜', '厨余垃圾_饼干', '厨余垃圾_香蕉皮',
           '厨余垃圾_骨肉相连', '厨余垃圾_鸡翅', '可回收物_乒乓球拍', '可回收物_书', '可回收物_保温杯', '可回收物_保鲜盒',
           '可回收物_信封', '可回收物_充电头', '可回收物_充电宝', '可回收物_充电线', '可回收物_八宝粥罐', '可回收物_刀',
           '可回收物_剃须刀片', '可回收物_剪刀', '可回收物_勺子', '可回收物_单肩包手提包', '可回收物_卡', '可回收物_叉子',
           '可回收物_变形玩具', '可回收物_台历', '可回收物_台灯', '可回收物_吹风机', '可回收物_呼啦圈', '可回收物_地球仪',
           '可回收物_地铁票', '可回收物_垫子', '可回收物_塑料瓶', '可回收物_塑料盆', '可回收物_奶盒', '可回收物_奶粉罐',
           '可回收物_奶粉罐铝盖', '可回收物_尺子', '可回收物_帽子', '可回收物_废弃扩声器', '可回收物_手提包', '可回收物_手机',
           '可回收物_手电筒', '可回收物_手链', '可回收物_打印机墨盒', '可回收物_打气筒', '可回收物_护肤品空瓶', '可回收物_报纸',
           '可回收物_拖鞋', '可回收物_插线板', '可回收物_搓衣板', '可回收物_收音机', '可回收物_放大镜', '可回收物_易拉罐',
           '可回收物_暖宝宝', '可回收物_望远镜', '可回收物_木制切菜板', '可回收物_木制玩具', '可回收物_木质梳子', '可回收物_木质锅铲',
           '可回收物_枕头', '可回收物_档案袋', '可回收物_水杯', '可回收物_泡沫盒子', '可回收物_灯罩', '可回收物_烟灰缸', '可回收物_烧水壶',
           '可回收物_热水瓶', '可回收物_玩偶', '可回收物_玻璃器皿', '可回收物_玻璃壶', '可回收物_玻璃球', '可回收物_电动剃须刀', '可回收物_电动卷发棒',
           '可回收物_电动牙刷', '可回收物_电熨斗', '可回收物_电视遥控器', '可回收物_电路板', '可回收物_登机牌', '可回收物_盘子', '可回收物_碗',
           '可回收物_空气加湿器', '可回收物_空调遥控器', '可回收物_纸牌', '可回收物_纸箱', '可回收物_罐头瓶', '可回收物_网卡', '可回收物_耳套',
           '可回收物_耳机', '可回收物_耳钉耳环', '可回收物_芭比娃娃', '可回收物_茶叶罐', '可回收物_蛋糕盒', '可回收物_螺丝刀', '可回收物_衣架',
           '可回收物_袜子', '可回收物_裤子', '可回收物_计算器', '可回收物_订书机', '可回收物_话筒', '可回收物_购物纸袋', '可回收物_路由器',
           '可回收物_车钥匙', '可回收物_量杯', '可回收物_钉子', '可回收物_钟表', '可回收物_钢丝球', '可回收物_锅', '可回收物_锅盖', '可回收物_键盘',
           '可回收物_镊子', '可回收物_鞋', '可回收物_餐垫', '可回收物_鼠标', '有害垃圾_LED灯泡', '有害垃圾_保健品瓶', '有害垃圾_口服液瓶', '有害垃圾_指甲油',
           '有害垃圾_杀虫剂', '有害垃圾_温度计', '有害垃圾_滴眼液瓶', '有害垃圾_玻璃灯管', '有害垃圾_电池', '有害垃圾_电池板', '有害垃圾_碘伏空瓶', '有害垃圾_红花油',
           '有害垃圾_纽扣电池', '有害垃圾_胶水', '有害垃圾_药品包装', '有害垃圾_药片', '有害垃圾_药膏', '有害垃圾_蓄电池', '有害垃圾_血压计']

label_data3_part=['厨余垃圾_白菜','可回收物_塑料瓶','其他垃圾_口罩','有害垃圾_电池']
label_data3 = np.array(label_data3_all)
model_data3="trash_data3_AlexNet"
test_path_data3='../datasets/data3/TEST/'

def create_pre_csv(csv_name):
    path = csv_name+"_predict_result.csv"
    with open(path,'w',encoding='utf8',newline="") as f:
        write = csv.writer(f)
        head = ["Label","Size","Period","Accuracy","Run Rate"]
        write.writerow(head)
        return path

class AlexNetModelTest:
    def __init__(self,test_path,label,model_name,imgsize=(128,128)):
        self.size=0
        self.period=0
        self.acc=0
        self.run_rate=0
        self.test_path=test_path
        self.label=label
        self.model_name=model_name
        self.model=load_model('../model/'+model_name+'.h5')
        self.imgsize=imgsize

    def predict_one_image(self,img_path):

        image = load_img(img_path)
        image = image.resize(self.imgsize)
        image = img_to_array(image)
        image = image/255
        image = np.expand_dims(image, 0)
        result = self.label[self.model.predict_classes(image)]
        return result

    def predict_dir_image(self,img_dir,label_name):
        imgs_name = os.listdir(img_dir)
        self.size=0
        hit_num=0
        print('########## Test Label: {} ##########'.format(label_name))
        start = time.process_time()
        for name in imgs_name:
            img_name = img_dir + name
            re = self.predict_one_image(img_name)
            self.size=self.size+1
            print(name+"：{}".format(re[0]))
            if re == label_name:
                hit_num = hit_num+1
        self.acc=hit_num/self.size
        #print("acc: {}".format(acc))
        end = time.process_time()
        self.period=end-start
        self.run_rate=self.period/self.size
        #print("本次预测一共运行了:%s秒----约等于%s分钟" % (time_period, time_period/60))
        #print("单张:%s秒" % speed)

    def img_test(self):
        csv_path=create_pre_csv(self.model_name)
        with open(csv_path,'a+',encoding='utf8',newline="") as csv_file:
            csv_write = csv.writer(csv_file)
            for i in self.label:
                self.predict_dir_image(self.test_path +i+'/',i)
                csv_write.writerows([(i,self.size,self.period,self.acc,self.run_rate)])

    def video_test(self,camera_index):
        camera = cv2.VideoCapture(camera_index)
        n = 0
        while True:
            ret, frame = camera.read()
            if not ret:
                continue
            cv2.imshow('frame', frame)
            n = n + 1

            if n % 10 == 0:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = frame.resize((128, 128))
                frame = img_to_array(frame)
                frame = frame/255
                frame = np.expand_dims(frame, 0)
                print(self.label[self.model.predict_classes(frame)])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model=input("please input model(data1/data2/data3): ")
    if model=='data1':
        modelTest=AlexNetModelTest(test_path_data1,label_data1,model_data1,(128,128))
    if model=='data2':
        modelTest=AlexNetModelTest(test_path_data2,label_data2,model_data2)
    elif model=='data3':
        modelTest=AlexNetModelTest(test_path_data3,label_data3,model_data3)
    mode=input("please input model test way(img/video): ")
    if mode=='img':

        modelTest.img_test()
        '''
        if model=='data2':
            modelTest.img_test()
        elif model=='data3':
            csv_path=create_pre_csv(modelTest.model_name)
            with open(csv_path,'a+',encoding='utf8',newline="") as csv_file:
                csv_write = csv.writer(csv_file)
                for i in label_data3_part:
                    modelTest.predict_dir_image(modelTest.test_path +i+'/',i)
                    csv_write.writerows([(i,modelTest.size,modelTest.period,modelTest.acc,modelTest.run_rate)])
        '''
    elif mode=='video':
        modelTest.video_test(0)
    print('model test end')


