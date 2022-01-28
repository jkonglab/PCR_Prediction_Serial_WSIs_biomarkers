
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from tqdm import tqdm
from glob import glob
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from datetime import datetime
import time

import numpy as np
import resnet
import cv2

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"


lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
#csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 32
nb_classes = 2
nb_epoch = 200


# input image dimensions
img_rows, img_cols = 512, 512
# The CIFAR10 images are RGB.
img_channels = 9






### random split training and testing based on patient id
patient=set()
for item in glob('../image_train/*/*HES*'):
    pcrflag,pid=item.split('/')[2],item.split('/')[-1].split('_')[0]
    patient.add((pcrflag,pid))

testindex=np.random.choice(range(73),30,replace=False)
trainindex=list(set(range(73))-set(testindex))
train_p=np.array(list(patient))[trainindex]
test_p=np.array(list(patient))[testindex]

np.save('train_p.npy',train_p)
np.save('test_p.npy',test_p)

X=[]
Y=[]
for item in tqdm(train_p):
    num=len(glob('../image_train/{}/{}_*HES*'.format(item[0],item[1])))
    if item[0]=='pcr':
        y=np.zeros((num,2))
        y[:,0]=1
    elif item[0]=='nonpcr':
        y=np.zeros((num,2))
        y[:,1]=1
    Y.append(y)
    for imgfile in glob('../image_train/{}/{}_*HES*'.format(item[0],item[1])):
        imghe=np.array(np.load(imgfile))
        imghe=cv2.resize(imghe,(512,512)).reshape((1,512,512,3))
        imgki=np.array(np.load(imgfile.replace('HES','KI-67')))
        imgki=cv2.resize(imgki,(512,512)).reshape((1,512,512,3))
        imgphh=np.array(np.load(imgfile.replace('HES','PHH3')))
        imgphh=cv2.resize(imgphh,(512,512)).reshape((1,512,512,3))
        X.append(np.concatenate([imghe,imgki,imgphh]))

X=np.concatenate(X,axis=0)
Y=np.concatenate(Y,axis=0)
print(X.shape,Y.shape)

X_train=X
Y_train=Y


X=[]
Y=[]
for item in tqdm(test_p):
    num=len(glob('../image_train/{}/{}_*HES*'.format(item[0],item[1])))
    if item[0]=='pcr':
        y=np.zeros((num,2))
        y[:,0]=1
    elif item[0]=='nonpcr':
        y=np.zeros((num,2))
        y[:,1]=1
    Y.append(y)
    for imgfile in glob('../image_train/{}/{}_*HES*'.format(item[0],item[1])):
        imghe=np.array(np.load(imgfile))
        imghe=cv2.resize(imghe,(512,512)).reshape((1,512,512,3))
        imgki=np.array(np.load(imgfile.replace('HES','KI-67')))
        imgki=cv2.resize(imgki,(512,512)).reshape((1,512,512,3))
        imgphh=np.array(np.load(imgfile.replace('HES','PHH3')))
        imgphh=cv2.resize(imgphh,(512,512)).reshape((1,512,512,3))
        X.append(np.concatenate([imghe,imgki,imgphh]))


X=np.concatenate(X,axis=0)
Y=np.concatenate(Y,axis=0)
print(X.shape,Y.shape)

X_test=X
Y_test=Y        



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.
print(X_train.shape,Y_train.shape)
print(np.max(X_train),np.min(X_train))

model = resnet.ResnetBuilder.build_resnet_18((9, 512, 512), 2)
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper])


