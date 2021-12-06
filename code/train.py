'''
Author: Ken Kaneki
Date: 2021-12-04 18:13:42
LastEditTime: 2021-12-05 18:42:40
Description: README
'''
#coding:utf-8
import os
import time

from PIL import ImageFile
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.layers.pooling import MaxPooling2D

ImageFile.LOAD_TRUNCATED_IMAGES = True

#仅取卷积层和池化层，去掉最后一层全连接层
# MobileNet_model = MobileNet(weights = 'imagenet', include_top = False, input_shape=(128,128,3))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def AlexNet(label_num):
    model = Sequential()
    # input_shape = (64,64, self.config.channles)
    input_shape = input_shape=(128,128,3)
    model.add(Convolution2D(64, (11, 11), input_shape=input_shape,strides=(1, 1),  padding='valid',activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))#26*26
    model.add(Convolution2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Convolution2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Convolution2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Convolution2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(label_num, activation='softmax'))
    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def AlexNet_Train(datasets_train_path,datasets_test_path,model_save_path):
    start = time.process_time()

    train_datagen = ImageDataGenerator(
    rotation_range = 10, #随机旋转度数
    width_shift_range = 0.1, #随机水平平移
    height_shift_range = 0.1,#随机竖直平移
    rescale = 1/255, #数据归一化
    shear_range = 0.1, #随机裁剪
    zoom_range = 0.1, #随机放大
    horizontal_flip = True, #水平翻转
    fill_mode = 'nearest', #填充方式
    )

    test_datagen = ImageDataGenerator(
    rescale = 1/255, #数据归一化
    )
    batch_size = 32

    #生成训练数据
    train_generator = train_datagen.flow_from_directory(
    datasets_train_path,
    target_size = (128,128),
    batch_size = batch_size,
    )
    #生成测试数据
    test_generator = test_datagen.flow_from_directory(
    datasets_test_path,
    target_size = (128,128),
    batch_size = batch_size,
    )
    print(len(train_generator.class_indices))

    model_AlexNet = AlexNet(len(train_generator.class_indices))
    model_AlexNet.summary()#输出模型网络结构
    filepath = model_save_path
    # Callbacks
    callbacks_list = [
    ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True,mode='max',period=1)]
    #定义优化器，代价函数，训练过程中计算准确率
    model_AlexNet.compile(optimizer = SGD(lr=1e-4,momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model_AlexNet.fit_generator(train_generator,epochs=5000,validation_data=test_generator, callbacks=callbacks_list)
    print(" model is save successfuly!")
    end = time.process_time()
    print("本次训练一共运行了:%s秒----约等于%s分钟"%((end-start), (end-start)/60))

if __name__ == '__main__':

    # data3
    #train_path='../datasets/data3/TRAIN'
    #test_path='../datasets/data3/TEST'
    #model_path='../model/trash_data3_AlexNet_new.h5'

    # data1
    train_path='../datasets/data1/TRAIN'
    test_path='../datasets/data1/TEST'
    model_path='../model/trash_data1_AlexNet.h5'

    AlexNet_Train(train_path,test_path,model_path)
