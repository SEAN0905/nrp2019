# this file is to write TCNNP based GAN using dataset based on text label
# TCNNP: transposed convolution neural network privatizer

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import keras
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Input, Reshape, LeakyReLU, Lambda, Add
from keras.layers import Concatenate, GaussianNoise, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam

base_path = "face32_relabeled/"


def pixel_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def pixel_mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


class GAP():
    def __init__(self):

        optimizer = SGD(lr=0.01, momentum=0.9)

        self.generator = self.build_generator()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # only generator will be trained
        self.discriminator.trainable = False

        img_raw = Input(shape=(1024, 1))
        noise = Input(shape=(4, 4, 256))
        img_prv = self.generator([img_raw, noise])
        clasif_res = self.discriminator(img_prv)

        # weight of adversary loss
        self.loss_x = 0.1

        self.combined = Model([img_raw, noise], [img_prv, clasif_res])

        # TODO: to be updated after FNNP model
        self.combined.compile(optimizer=optimizer, loss=[
                              pixel_mse_loss, "categorical_crossentropy"], loss_weights=[self.loss_x, -1])

    def read_data(self):
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
            data_raw = np.reshape(raw_image, (1024, ))
            for j, pixel_value in enumerate(data_raw):
                data_raw[j] = data_raw[j] / 255.0
            data_n = np.reshape(data_raw, (1024, 1))
            X_train.append(data_n)
            Y_gender.append([image_gender_label == "0",
                             image_gender_label == "1"])
            Y_smile.append([image_smile_label == "0",
                            image_smile_label == "1"])
        #     print(Y_gender[1:6])
        #     print(Y_smile[1:6])
        # # [[False, True], [False, True], [True, False], [True, False], [True, False]]
        # # [[False, True], [False, True], [False, True], [False, True], [False, True]]
        return np.asarray(X_train), np.asarray(Y_gender), np.asarray(Y_smile)

    def build_generator(self):
        # the function to initialize a privatizer(generator)
        # input: random noise generated outside function
        # output: processed noise to be added onto images

        # TODO: supposedly tf.shape(noise_input) is (4, 4, 256)
        # Yet actual result is (4, )
        # noise_input = Input(shape=(4, 4, 256))
        noise_input = Input(shape=(4096, 1))
        noise_reshape = Reshape((256, 4, 4))(noise_input)
        print(noise_reshape)

        deconv1 = Conv2DTranspose(filters=128, kernel_size=(
            3, 3), strides=2, activation="relu")(noise_reshape)
        batch1 = BatchNormalization()(deconv1)
        print(batch1)
        deconv2 = Conv2DTranspose(filters=16, kernel_size=(
            3, 3), strides=2, activation="tanh")(batch1)
        batch2 = BatchNormalization()(deconv2)
        print(batch2)
        deconv3 = Conv2DTranspose(filters=1, kernel_size=(
            3, 3), strides=2, activation="tanh")(batch2)
        batch3 = BatchNormalization()(deconv3)
        print(batch3)
        batch3_reshape = Reshape((1024, 1), input_shape=(32, 32, 1))(batch3)

        img_input = Input(shape=(1024, 1))
        img_input_n = BatchNormalization()(img_input)
        img_input_dense = Dense(1)(img_input_n)

        result = Add()([img_input_dense, batch3_reshape])

        final = Reshape((32, 32, 1))(result)

        return Model([noise_input, img_input], final)

    def build_discriminator(self):
        # to initialize the pre-trained discriminator
        # input: privatized image from generator
        # output: classification result

        model = Sequential([
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding='same',
                activation='relu',
                input_shape=(32, 32, 1)),
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

        model.summary()

        model.load_weights("adversary32.h5")

        img_prv = Input(shape=self.img_shape)
        clasify_res = model(img_prv)

        return Model(img_prv, clasify_res)

    def train(self, epochs, batch_size=64, sample_interval=50):
        # load raw data
        X_data_raw, Y_gender_raw, Y_smile_raw = self.read_data()

        # print(X_data_raw.shape, Y_gender_raw.shape, Y_smile_raw.shape)
        # # (2723, 1024) (2723, 2) (2723, 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # --------------------

            # select random batch of images
            ids = np.random.randint(0, 2723-1, batch_size)
            imgs = X_data_raw[ids]
            gender_label = Y_gender_raw[ids]
            smile_label = Y_smile_raw[ids]

            # parameter for random noise
            self.mu, self.sigma = 0, 0.1
            # TODO: find proper way for linear projection
            noise = tf.random.normal((4, 4, 256), self.mu, self.sigma)

            # generate privatized images
            prv_imgs = self.generator(
                [imgs, np.random.normal(0, 1, (batch_size, 4, 4, 256))])

            # train the discriminator
            d_loss_prv = self.discriminator.train_on_batch(
                prv_imgs, gender_label)
            # print(d_loss_prv)

            # update penalty coefficient
            self.loss_x = epoch * 0.01

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(
                [imgs, np.random.normal(0, 1, (batch_size, 4, 4, 256))], [imgs, gender_label])
            print()
            print("loss_x: %.1f" % self.loss_x)
            print("Epoch %d [D loss: %.5f, acc. : %.3f %%] [G loss: combined: %.5f; pixel_mse_loss: %.5f; categorical_crossentropy: %.5f]" % (
                epoch, d_loss_prv[0], 100*d_loss_prv[1], g_loss[0], g_loss[1], g_loss[2]))
            print()


if __name__ == "__main__":
    gap = GAP()
    gap.train(epochs=20)
