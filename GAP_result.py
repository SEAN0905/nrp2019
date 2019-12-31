from keras.optimizers import SGD
from keras.models import Sequential, Model, load_model
from keras.losses import categorical_crossentropy
from keras.activations import tanh
from keras.layers import concatenate, Input, Reshape, LeakyReLU, Lambda, Concatenate, GaussianNoise
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import models, regularizers, optimizers, backend as K
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)


dataset_path = "face32_relabeled/"


def pixel_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def pixel_mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


optimizer = SGD(lr=0.05, momentum=0.9)

path_model_to_eval = "H5_file/GAN_FNNP_0.02_4_loss.h5"

combined_model = load_model(path_model_to_eval)
combined_model.compile(optimizer=optimizer, loss=[
               pixel_mse_loss, "categorical_crossentropy"], loss_weights=[2.1, 1])


def read_data(self):
    # gender.txt: 0 for woman, 1 for man
    handle1 = open(dataset_path + "gender.txt", "r")
    # smile.txt: 0 for non-smile, 1 for smile
    handle2 = open(dataset_path + "smile.txt", "r")

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
        image_name = dataset_path + "image/" + str(i) + ".jpg"
        try:
            image = Image.open(image_name)
            image.load()
        except:
            print("error1", i)
        continue
        image_gender_label = raw_gender_label[i-1]
        image_smile_label = raw_smile_label[i-1]
        raw_image = np.asarray(image, dtype="int32")
        # print(raw_image.shape)
        data_raw = np.reshape(raw_image, (1024, ))
        # print(data_raw.shape)
        for j, pixel_value in enumerate(data_raw):
            data_raw[j] = data_raw[j] / 255.0
            data_n = np.reshape(data_raw, (32, 32, 1))
            X_train.append(data_n)
            Y_gender.append([image_gender_label == "0",
                     image_gender_label == "1"])
            Y_smile.append([image_smile_label == "0",
                    image_smile_label == "1"])
        # print(Y_gender[1:6])
        # print(Y_smile[1:6])
        # # [[False, True], [False, True], [True, False], [True, False], [True, False]]
        # # [[False, True], [False, True], [False, True], [False, True], [False, True]]
    return np.asarray(X_train), np.asarray(Y_gender), np.asarray(Y_smile)

X_data_raw, Y_gender, Y_smile = read_data()
score = combined_model.evaluate([X_data_raw, np.random(0, 1, (2723, 100, 1))], [X_data_raw, Y_gender])
print("score", score)