#%%
from keras import layers
from keras import models
#import keras.backend.tensorflow_backend as keras
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))``
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#%%
model.summary()

#%%
from keras import optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])

#%%

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

#%%
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

#%%

from keras.callbacks import ModelCheckpoint
import os

model_name = 'keras_{epoch:02d}-{val_loss:.4f}.h5'

model_path = os.path.join('/home/ha/codes/keras/models',model_name)

cb_checkpoint = ModelCheckpoint(filepath=model_path,monitor='val_loss'
,verbose=1, save_best_only=False, period=10)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,\
    validation_data=validation_generator,
    validation_steps=50,
    callbacks= [cb_checkpoint]
)


#%%
model.load_weights("/home/ha/codes/keras/models/keras_100-0.4246.h5")

#%%

import matplotlib.pyplot as plt


#%%
