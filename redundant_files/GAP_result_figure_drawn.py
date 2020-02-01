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

def build_generator():
        # the function to initialize a privatizer(generator)
        # input: raw image
        # output: privatized image

        img_input = Input(shape=(32, 32, 1))
        img_input_reshape = Reshape((1024, 1))(img_input)
        img_input_reshape_n = BatchNormalization()(img_input_reshape)
        img_input_dense = Dense(1)(img_input_reshape_n)

        noise = Input(shape=(100, 1))
        noise_dense = Dense(1)(noise)
        img_cat = Concatenate(axis=1)([img_input_dense, noise_dense])
        # print("img_cat", img_cat)
        # # (?, 1124, 1)

        receiver = Flatten()(img_cat)
        dense1 = Dense(1024)(receiver)
        dense1_a = LeakyReLU()(dense1)
        dense2 = Dense(1024)(dense1_a)
        dense2_a = LeakyReLU()(dense2)
        dense3 = Dense(1024)(dense2_a)
        dense3_a = LeakyReLU()(dense3)
        dense4 = Dense(1024)(dense3_a)
        dense4_a = Activation("tanh")(dense4)

        final = Lambda(lambda x: x+1)(dense4_a)
        # print("final: ", final)
        # # (?, 1024)
        img_prv = Reshape((32, 32, 1), input_shape=(1024, ))(final)
        # print("img_prv", img_prv)
        # # (?, 32, 32, 1)

        return Model([img_input, noise], img_prv)

def build_discriminator():
        # to initialize the pre-trained discriminator
        # input: privatized image from generator
        # output: classification result

        model = Sequential([
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu', input_shape=(32, 32, 1)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3),
                   padding='same', activation='relu'),
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

        # model.summary()

        model.load_weights("adversary32.h5")

        img_prv = Input(shape=(32, 32, 1))
        clasify_res = model(img_prv)

        return Model(img_prv, clasify_res)

discriminator = build_discriminator()
generator = build_generator()

z = Input(shape=(32, 32, 1))
noise = Input(shape=(100, 1))
img_prv = generator([z, noise])
clasif_res = discriminator(img_prv)
combined_model = Model([z, noise], [img_prv, clasif_res])

optimizer = SGD(lr=0.05, momentum=0.9)
path_model_to_eval = "mse0.005_acc53_weights.h5"

combined_model.summary()
combined_model.load_weights(path_model_to_eval)

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