# the file is to write a adversary model based on image data generator
# the dataset used is the manually classified face64

import numpy as np
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# base_path = "../dataset/face64"
base_path = "face64/image"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)

# should create separate y_train instead
train_generator = train_datagen.flow_from_directory(
    base_path,
    color_mode="grayscale",
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True)

# print(train_generator[0][0].shape)

validation_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(64, 64),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=True)

# supposed to convert to a binary classification result
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

# print(train_generator[0][1])

model = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dense(1024, kernel_initializer='random_uniform',
        # ########### change next line's l2 parameter to tune
          activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    BatchNormalization(),
    Dense(1024, kernel_initializer='random_uniform',
        # ########### change next line's l2 parameter to tune
          activation='relu', kernel_regularizer=regularizers.l2(0.2)),
    BatchNormalization(),
    Flatten(),
    # supposed to be 2 results for prediction, one predicting smile and the other not
    Dense(2, activation='softmax')
])

model.load_weights("adversary_gen_overall.h5")

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

model.summary()

callbacks = [
    EarlyStopping(monitor='val_acc', patience=200, verbose=1),
    ModelCheckpoint(filepath="adversary_64_gen_checkpoint.h5py",
                    monitor='val_acc',
                    save_best_only=True,
                    verbose=1)
]

history = model.fit_generator(
    train_generator,
    # update based on change in batch size
    steps_per_epoch=3571 // 32,
    validation_steps=3571 // 32,
    epochs=30,
    validation_data=validation_generator,
    callbacks=callbacks)

print(history.history.keys())
model.save_weights("adversary_gen_overall.h5")
