# the accuracy of the classifier: 
# train

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers, optimizers
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import logging
import numpy as np
import os
import random
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)

base_path = "face32_gen/image_smile"

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)

# should create separate y_train instead
train_generator = train_datagen.flow_from_directory(
    base_path,
    color_mode="grayscale",
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True)

# print(train_generator[0][0].shape)

validation_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(32, 32),
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
        input_shape=(32, 32, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dense(1024, kernel_initializer='random_uniform',
          # ########### change next line's l2 parameter to tune
          activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    BatchNormalization(),
    Dense(1024, kernel_initializer='random_uniform',
          # ########### change next line's l2 parameter to tune
          activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    BatchNormalization(),
    Flatten(),
    Dense(2, activation='softmax')
])

# model.load_weights("smile_overall.h5")

sgd = optimizers.SGD(lr=0.00008, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

# model.summary()

callbacks = [
    EarlyStopping(monitor='val_acc', patience=200, verbose=1),
    ModelCheckpoint(filepath="smile_checkpoint.h5py",
                    monitor='val_acc',
                    save_best_only=True,
                    verbose=1)
]

history = model.fit_generator(
    train_generator,
    # # update based on change in batch size
    steps_per_epoch=2822 // 32,
    validation_steps=2822 // 32,
    epochs=19,
    validation_data=validation_generator,
    callbacks=callbacks)

print(history.history.keys())
key_list = ['val_loss', 'val_acc', 'loss', 'acc']
for i in range(4):
    current_key = key_list[i]
    print("\n")
    print(current_key)
    for epoch_number in range(19):
        print(str(epoch_number+1) +  ", " + str(history.history[current_key][epoch_number]))
        
# # model.save_weights("smile_overall.h5")
# plt.gca().set_ylim([0.5, 1])
# plt.plot(history.history["acc"])
# plt.plot(history.history["val_acc"])
# plt.title("model accuracy")
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.show()
# plt.savefig("Classifier_smile_accuracy.png")
# plt.close()

# plt.gca().set_ylim([0, 0.7])
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.show()
# plt.savefig("Classifier_loss.png")
# plt.close()
