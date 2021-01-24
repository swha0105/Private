#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pixiedust
import os
import cv2
import numpy as np
import matplotlib.pylab as plt
#import nibabel as nib
#import PIL
#import pandas as pd1
import random
import pydicom
from scipy.ndimage.interpolation import rotate
from skimage.io import imsave, imread
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

import copy
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras import backend as K
from keras import losses
import tensorflow as tf
from keras.models import load_model
import argparse


# In[2]:


## model 3D Unet
def get_unet():
    
    n = 8
    
    inputs = Input((None, None,  None, 1))
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv3D(n, (3, 3, 3), padding='same')(conv1)
    conv1 = BatchNormalization(axis=4)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = Conv3D(n, (3, 3, 3), padding='same', strides = (2 ,2, 2))(conv1)
    pool1 = BatchNormalization(axis=4)(pool1)
    pool1 = Activation('relu')(pool1)
#    pool1 = Dropout(0.25)(pool1)
#    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv22 = concatenate([pool1, MaxPooling3D(pool_size=(2, 2, 2))(inputs)], axis=4)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv22)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv3D(2*n, (3, 3, 3), padding='same')(conv2)
    conv2 = BatchNormalization(axis=4)(conv2)
    conv2 = Activation('relu')(conv2)                           
    pool2 = Conv3D(2*n, (3, 3, 3), padding='same', strides = (2, 2, 2))(conv2)
    pool2 = BatchNormalization(axis=4)(pool2)
    pool2 = Activation('relu')(pool2)
#    pool2 = Dropout(0.25)(pool2)
#    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv33 = concatenate([pool2, MaxPooling3D(pool_size=(2, 2, 2))(conv22)], axis=4)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv33)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)    
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)
    conv3 = BatchNormalization(axis=4)(conv3)
    conv3 = Activation('relu')(conv3)
#    conv3 = Conv3D(4*n, (3, 3, 3), padding='same')(conv3)
#    conv3 = BatchNormalization(axis=4)(conv3)
#    conv3 = Activation('relu')(conv3)
    pool3 = Conv3D(2*n, (3, 3, 3), padding='same', strides = (2, 2, 2))(conv3)
    pool3 = BatchNormalization(axis=4)(pool3)
    pool3 = Activation('relu')(pool3)
    
    
    conv44 = concatenate([pool3, MaxPooling3D(pool_size=(2, 2, 2))(conv33)], axis=4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv44)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(concatenate([conv44, conv4], axis=4))
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv3D(8*n, (3, 3, 3), padding='same')(conv4)    
    conv4 = BatchNormalization(axis=4)(conv4)
    conv4 = Activation('relu')(conv4)
#    conv3 = Dropout(0.5)(conv3)
#    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
#    conv4 = BatchNormalization(axis=1)(conv4)
#    conv4 = Activation('relu')(conv4)
#    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
#    conv4 = BatchNormalization(axis=1)(conv4)

    up5 = concatenate([Conv3DTranspose(2*n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(up5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(conv5)
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv3D(4*n, (3, 3, 3), padding='same')(concatenate([up5, conv5], axis=4))
    conv5 = BatchNormalization(axis=4)(conv5)
    conv5 = Activation('relu')(conv5)
    
    up6 = concatenate([Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv5), conv2], axis=4)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(up6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(conv6)
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv3D(2*n, (3, 3, 3), padding='same')(concatenate([up6, conv6], axis=4))
    conv6 = BatchNormalization(axis=4)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv3DTranspose(n, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv6), conv1], axis=4)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(up7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(conv7)
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv3D(n, (3, 3, 3), padding='same')(concatenate([up7, conv7], axis=4))
    conv7 = BatchNormalization(axis=4)(conv7)
    conv7 = Activation('relu')(conv7)
 #   up6 = concatenate([Conv3DTranspose(16, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv5), conv1], axis=4)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
 #   conv6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv6)
 #   conv6 = BatchNormalization(axis=1)(conv6)
 #   conv6 = Activation('relu')(conv6)
    
    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)
    
    model = Model(inputs=[inputs], outputs=[conv8])
    
    learning_rate = 1e-3
    decay_rate = learning_rate/300
    
    model.compile(optimizer=Adam(lr=learning_rate, decay=decay_rate), loss=dice_coef_loss, metrics=[dice_coef])
#   model.compile(optimizer=Adam(lr=1e-5), loss=losses.mean_squared_error, metrics=[dice_coef])

    return model


# In[ ]:


## loss function
smooth = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth * 0.01) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[ ]:


# def preprocess(images, rows=img_rows, cols=img_cols, slices=img_slice):
#     imgs_p = np.ndarray((images.shape[0], rows, cols, slices), dtype=np.float32)
#     for i in range(images.shape[0]):
#         imgs_p[i] = resize(images[i], (rows, cols, slices), preserve_range=True)

#     imgs_p = imgs_p[..., np.newaxis]
#     return imgs_p


# In[ ]:


def convertDicomToImg(img_array, center, width, slope, intercept):
	cvtd_array = np.zeros(img_array.shape, dtype=np.uint8)
    
	temp = img_array * slope + intercept
	lth = center - 0.5 - (width-1)/2
	hth = center - 0.5 + (width-1)/2
	cvtd_array = 255*((temp-(center-0.5))/(width-1)+0.5)
	cvtd_array[temp<=lth] = 0
	cvtd_array[temp>hth] = 255

	return(cvtd_array)


# In[3]:


CT_path = 'G:\\JBS-05K\\2차\\CT'
ROI_path = 'G:\\JBS-05K\\2차\\ROI'


# In[4]:


pat_id_list = os.listdir(CT_path)
pat_id_list.sort()
PN = len(pat_id_list)
ROI_train = os.listdir(ROI_path)
ROI_train.sort()
PN_train = len(ROI_train)


# In[5]:


pname = 'DUM_' + ROI_train[0][0:4]
dcm_path = os.path.join(CT_path, pname)

label_path = os.path.join(ROI_path)
label_list = os.listdir(label_path)
label_list.sort()

dcm_list = os.listdir(dcm_path)
dcm_list.sort()

print(label_list)


# In[6]:


img_rows = 192
img_cols = 192
img_slice = 80

CT_mask = np.zeros((PN_train, img_rows, img_cols, img_slice), dtype=np.float32)
ROI_mask = np.zeros((PN_train, img_rows, img_cols, img_slice), dtype=np.float32)


# In[ ]:


label_path = os.path.join(ROI_path,ROI_train[0])


# In[91]:


print(label_path)
sname = 'slice-' + str(17) + '.jpg'
label = imread(os.path.join(label_path, sname))

label_test = label
label_test = label_test.sum(axis=2)/(3*255)

#print(np.histogram(label_test))

label_mean = 0.5
plt.figure(figsize=[20,10])
plt.subplot(121).imshow(label_test)

#label_test = np.array(label_test)
label_test[label_test>label_mean] = 255
label_test[label_test<label_mean] = 0
plt.subplot(122).imshow(label_test)
plt.show()
#print(np.histogram(label_test))


# In[ ]:


for i in range(PN_train):

    pname = 'DUM_' + ROI_train[i][0:4]
    
    dcm_path = os.path.join(CT_path, pname)
    dcm_list = os.listdir(dcm_path)
    dcm_list.sort()
    
    label_path = os.path.join(ROI_path,ROI_train[i])
    label_list = os.listdir(label_path)
    label_list.sort()

    
    DN = len(dcm_list)
    gap = img_slice - DN
    sgap = int(np.floor(gap / 2))

    for j in range(DN):

        image_path = os.path.join(dcm_path, dcm_list[j])

        dcm_info = pydicom.dcmread(image_path,force=True)
        if 'DERIVED' in dcm_info.ImageType:
            continue
        try:
            pixel = dcm_info.pixel_array
        except:
            continue
        try:
            image = convertDicomToImg(pixel, center=dcm_info.WindowCenter, width=dcm_info.WindowWidth, slope=dcm_info.RescaleSlope, intercept=dcm_info.RescaleIntercept)
        except:
            image = convertDicomToImg(pixel, center=dcm_info.WindowCenter[0], width=dcm_info.WindowWidth[0], slope=dcm_info.RescaleSlope, intercept=dcm_info.RescaleIntercept)

        sname = 'slice-' + str(j+1) + '.jpg'

    
        if sgap+j < 80:
            try:
                CT_mask[i,:,:,sgap+j]=cv2.resize(np.float32(image),(img_rows,img_cols))
            except:
                continue
        
        if sname in label_list:
            
            label = imread(os.path.join(label_path, sname))

            label_test = label
            label_test = label_test.sum(axis=2)/(3*255)

            print(np.histogram(label_test))

            #label_mean = np.mean(label_test)
            label_mean = 0.5
            #print(label_test.shape)
            #print(label_mean)
            plt.figure(figsize=[20,10])
            plt.subplot(121).imshow(label_test)

            label_test = np.array(label_test)
            label_test[label_test>label_mean] = 255
            label_test[label_test<label_mean] = 0

            
#             label = imread(os.path.join(label_path, sname))
#             label = np.float32(label)
#             label = label.sum(axis=0)           
            
            #label = label/255
            
            if sgap+j < 80:
                    ROI_mask[i,:,:,sgap+j]=cv2.resize(label_test,(img_rows,img_cols))
            else:
                continue
                    
#ROI_mask[ROI_mask >= 0.5] = 1
#ROI_mask[ROI_mask < 0.5] = 0         


# In[ ]:


import pickle
with open('ROI_mask','wb') as f:
    pickle.dump(ROI_mask,f)
with open('CT_mask','wb') as f:
    pickle.dump(CT_mask,f)


# In[7]:


import pickle
with open('ROI_mask','rb') as f:
    ROI_mask = pickle.load(f)
with open('CT_mask','rb') as f:
    CT_mask = pickle.load(f)


# In[ ]:


CT_pre = CT_mask.copy()
CT_pre = CT_pre[0,:,:,40]

CT_pre_ori = CT_pre.copy()

CT_pre[CT_pre>250] = 0
CT_pre[CT_pre==0] = 0 


# In[8]:


CT_pre = normalization(CT_pre)


# In[ ]:


for i in range(80):
    print(i)
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10,10))
    ax0.imshow(CT_mask[0,:,:,i],cmap=plt.cm.bone)
    ax1.imshow(ROI_mask[0,:,:,i])


# In[ ]:


pname = ['DUM_' + label[0:4] for label in ROI_train]
save_path = '/home/user/notebooks/SW/save_folder/'
if not os.path.isdir(save_path):
os.mkdir(save_path)
CT_input = copy.deepcopy(CT_mask)
ROI_out = copy.deepcopy(ROI_mask)


# In[ ]:


temp_CT = CT_input[0,0,:,:,20]


# In[ ]:


temp_CT_rotated = rotate(temp_CT,10)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10,10))
ax0.imshow(temp_CT,cmap='gray')
ax1.imshow(temp_CT_rotated,cmap='gray')
#plt.imshow(CT_input[0,0,:,:,20])


# In[ ]:


def normalization(array):

    d0,d1,d2,d3 = array.shape
    array_flat = array.flatten()
    array_flat = [np.nan if x == 0 else x for x in array_flat]
    mean = np.nanmean(array_flat)
    std = np.nanstd(array_flat)


# In[115]:


def normalization(array):

    d0,d1,d2,d3 = array.shape
    array_flat = array.flatten()
    array_flat = [np.nan if x == 0 else x for x in array_flat]
    mean = np.nanmean(array_flat)
    std = np.nanstd(array_flat)
    
    array_flat = (array_flat-mean)/std
    array = array_flat.reshape(d0,d1,d2,d3)
    
    return array


# In[ ]:




