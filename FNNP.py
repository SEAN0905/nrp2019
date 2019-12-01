# this file is to write FNNP based GAN using image data generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
from keras.layers import concatenate, Input, Reshape, LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

base_path = "face32_relabeled/"


def read_data():
    # gender.txt: 0 for woman, 1 for man
    handle1 = open(base_path + "gender.txt", "r")
    # smile.txt: 0 for non-smile, 1 for smile
    handle2 = open(base_path + "smile.txt", "r")

    raw_gender_label = []
    for line in handle1:
        line = line.strip()
        if line == "0" or line == "1":
            raw_gender_label.append(line)
        else:
            print(line)
    handle1.close()

    raw_smile_label = []
    for line in handle2:
        line = line.strip()
        if line == "0" or line == "1":
            raw_smile_label.append(line)
        else:
            print(line)
    handle2.close()

    # supposed to contain 2723 images
    X_train, Y_gender, Y_smile = [], [], []
    for i in range(1, 2723+1):
        image_name = base_path + "image/" + str(i) + ".jpg"
        try:
            image = Image.open(image_name)
            image.load()
        except:
            print("error1", i)
            continue
        image_gender_label = raw_gender_label[i-1]
        image_smile_label = raw_smile_label[i-1]
        raw_image = np.asarray(image, dtype="int32")
        data = np.reshape(raw_image, (1024, ))
        X_train.append(data)
        Y_gender.append([image_gender_label == "0", image_gender_label == "1"])
        Y_smile.append([image_smile_label == "0", image_smile_label == "1"])
#     print(Y_gender[1:6])
#     print(Y_smile[1:6])
# # [[False, True], [False, True], [True, False], [True, False], [True, False]]
# # [[False, True], [False, True], [False, True], [False, True], [False, True]]
    return np.asarray(X_train), np.asarray(Y_gender), np.asarray(Y_smile)


X_data_raw, Y_gender, Y_smile = read_data()

# concatenate random noise to images
# TODO: should the size of random noise be increased accordingly?
mu, sigma = 0, 0.1
X_data = np.asarray([
    np.append(np.reshape(X, (1024, 1)), np.random.normal(mu, sigma, 100))
    for X in X_data_raw
])

# split into train and test subsets
validation_split = 0.1
train_num = int(len(X_data) * (1 - validation_split))

X_train = X_data[:train_num]
X_test = X_data[train_num:]
# print(X_train.shape)
# # 2450, 1124
y_gender_train = Y_gender[:train_num]
y_gender_test = Y_gender[train_num:]
y_smile_train = Y_smile[:train_num]
y_smile_test = Y_smile[train_num:]

X_train_raw = X_data_raw[:train_num]
X_test_raw = X_data_raw[train_num:]

# privatizer initialization
privatizer = Sequential([
    Dense(1124, input_shape=(1124, )),
    LeakyReLU(alpha=0.3),
    Dense(1024),
    LeakyReLU(alpha=0.3),
    Dense(1024),
    LeakyReLU(alpha=0.3),
    Dense(1024),
    LeakyReLU(alpha=0.3),
    Reshape((32, 32, 1), input_shape=(1024,))
])
privatizer.inputs = X_train

# # supposed to load weights saved
# privatizer.load_weights("privatizer_overall.h5")

for item in privatizer.layers:
    item.trainable = False

# privatizer.compile(optimizer=SGD(lr=0.1, momentum=0.9),
#                    loss=["categorical_crossentropy"])
# privatizer.summary()

# pre-trained adversary model
adversary = Sequential([
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
          activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    BatchNormalization(),
    Dense(1024, kernel_initializer='random_uniform',
          activation='relu', kernel_regularizer=regularizers.l2(0.00001)),
    BatchNormalization(),
    Flatten(),
    Dense(2, activation='softmax')
])
adversary.load_weights("adversary_gen_overall_32.h5")
adversary.trainable = False
adversary.compile(loss=['categorical_crossentropy'],
                  optimizer=SGD(lr=0.01, momentum=0.9), metrics=['acc'])

# combine privatizer and adversary to have GAP
GAP = Model(
    input=privatizer.input,
    output=[adversary(privatizer.output), privatizer.layers[-2].output])
GAP.summary()


def X_loss(y_true, y_predicted):
    return K.mean(K.square(y_true - y_predicted) / (255.0 * 255.0))


n_epoch = 30
lr = 0.001

num_iter = 10

N_layer_p = len(privatizer.layers)

while True:
    loss_x = 1
    # loss_x = float(input("Input the penalty parameter:\n"))

    # train the privatizer model and set adversary to be untrainable
    for item in GAP.layers[:N_layer_p]:
        item.trainable = True
    GAP.layers[-1].trainable = False
    print(X_test.shape)
    GAP.compile(optimizer=SGD(lr=lr, momentum=0.9), loss=[
                "categorical_crossentropy", X_loss], loss_weights=[1, loss_x],)
    GAP.summary()
    GAP.fit(x=X_train, y=[y_gender_train, X_train_raw], batch_size=60,
            epochs=n_epoch, validation_data=([X_test], [y_gender_test, X_test_raw]))
    
    num_iter -= 1
    if num_iter < 0:
        break

evaluate_metric = GAP.evaluate(x=X_data, y=[Y_smile, Y_gender])
print(evaluate_metric)
GAP.save_weights("GAN.h5")
