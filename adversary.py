# a santized version of adversary model to train the gender classifier beforehand
import numpy as np
from PIL import Image
import glob, os, random
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

dataset_base_path = "GENKI_4K_64/"


def read_data():
    # read raw label for gender and smile information
    handle = open(dataset_base_path + "gender_labels_for_4K.txt", "r")

    raw_gender_label = []
    for line in handle:
        line = line.strip()
        if line == "man" or line == "woman":
            raw_gender_label.append(line)
        else:
            print(line)
    handle.close()
    # # print(raw_gender_label[1230])
    # print("r", len(raw_gender_label))
    # # 3986

    x_train, Y_gender, Y_smile = [], [], []
    # supposed to be 1164 in the end, merely a counter for check
    broken_series = 0
    for i in range(1, 3986 + 1):
        image_name = dataset_base_path + "image/" + str(i) + ".jpg"
        try:
            image = Image.open(image_name)
            image.load()
        except:
            # print(image_name)
            broken_series += 1
            continue
        image_gender_label = raw_gender_label[i - 1]
        # print(image_gender_label, image_smile_label)
        raw_data = np.asarray(image, dtype="int32")
        data = np.reshape(raw_data, (64, 64, ))
        x_train.append(data)
        # note the result for categorical classification
        # is an array with only one element to be 0
        Y_gender.append(
            [image_gender_label == 'man', image_gender_label == 'woman'])
    # print(Y_gender[1:6])
    # print("b", broken_series)
    return np.asarray(x_train), np.asarray(Y_gender)


X_data, Y_gender = read_data()

# split into train and test dataset
test_split = 0.1
train_num = int(len(X_data) * (1 - test_split))

# 2539: 283
X_train = X_data[:train_num]
X_test = X_data[train_num:]

y_gender_train = Y_gender[:train_num]
y_gender_test = Y_gender[train_num:]

model = Sequential([
    Conv2D(filters=32,
           kernel_size=3,
           padding='same',
           activation='relu',
           input_shape=(64, 64, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dense(1024, kernel_initializer='random_uniform', activation='relu'),
    BatchNormalization(),
    Dense(1024, kernel_initializer='random_uniform', activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

model.summary()

# callbacks = [
#     EarlyStopping(monitor='val_acc', patience=200, verbose=1),
#     ModelCheckpoint(filepath="model_checkpoint1.h5py",
#                     monitor='val_acc',
#                     save_best_only=True,
#                     verbose=1)
# ]

# TODO:regularization needed
history = model.fit(
    x=X_train,
    y=y_gender_train,
    steps_per_epoch=3571 // 32,
    validation_steps=3571 // 32,
    epochs=100,
    validation_data=(X_test, y_gender_test),
    # callbacks=callbacks,
)

print(history.history.keys())
model.save_weights("adversary.h5")
