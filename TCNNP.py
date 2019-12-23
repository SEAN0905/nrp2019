# this file is to write TCNNP based GAN using dataset based on text label
# TCNNP: transposed convolution neural network privatizer

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
from keras.layers import Input, Reshape, LeakyReLU, Lambda
from keras.layers import Concatenate, GaussianNoise, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam

base_path = "face32_relabeled/"

# set global variables
global penalty_coef
global distortion
# set penalty term for privatizer loss function
penalty_coef = 0.01
# distortion to limit change to original image
distortion = 0.1


def privatizer_loss(y_true, y_pred):
    # standard procedure with numpy as inputy
    def _log_loss(input_tensors, eps=1e-15):
        # unpack input tensors
        _y_true, _y_pred = input_tensors
        # Prepare numpy array data
        _y_true = np.array(_y_true)
        # print(y_true.shape)
        _y_pred = np.array(_y_pred)
        # print(y_pred.shape)

        # clip the y_pred
        p = np.clip(_y_pred, eps, 1-eps)
        loss = np.sum(- _y_true * np.log(p) - (1 - _y_true) * np.log(1-p))
        log_loss = loss / len(_y_true)

        distortion_punishment = penalty_coef * max(0, np.mean(np.square(_y_true - _y_pred) - distortion))
        return log_loss + distortion_punishment
        
    # wrap python function as an operation in tensorflow graph
    return tf.numpy_function(_log_loss, [y_true, y_pred], tf.float32)

class GAP():
    def __init__(self):
        X_raw, Y_gender, Y_smile = self.read_data()

        # TODO: tune learning rate of optimizer
        optimizer = SGD(lr=0.01, momentum=0.9)

        # parameter for random noise
        self.mu, self.sigma = 0, 0.1
        # TODO: find proper way for linear projection
        noise = tf.random.normal((4, 4, 256), self.mu, self.sigma)

        self.generator = self.build_generator()

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        
        img_raw = Input(shape=(1024, 1))
        img_prv = tf.add(self.generator(tf.random.normal((4, 4, 251), self.mu, self.sigma)), img_raw)
        clasif_res = self.discriminator(img_prv)

        # weight of adversary loss
        self.loss_x = 0.1

        self.combined = Model(img_raw, [img_prv, clasif_res])

        # TODO: to be updated after FNNP model
        self.combined.compile(optimizer=optimizer, loss=[privatizer_loss, "categorical_crossentropy"], loss_weights=[1, self.loss_x])

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
            data = np.reshape(raw_image, (1024, ))
            X_train.append(data)
            Y_gender.append([image_gender_label == "0", image_gender_label == "1"])
            Y_smile.append([image_smile_label == "0", image_smile_label == "1"])
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
        noise_input = Input(shape=(4, 4, 256))
        print(tf.shape(noise_input))

        deconv1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation="relu", data_format="channels_last")(noise_input)
        batch1 = BatchNormalization()(deconv1)
        print(tf.shape(batch1))
        deconv2 = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation="tanh", data_format="channels_last")(batch1)
        batch2 = BatchNormalization()(deconv2)
        print(tf.shape(batch2))
        deconv3 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation="tanh", data_format="channels_last")(batch2)
        batch3 = BatchNormalization()(deconv3)
        print(tf.shape(batch3))
        result = Reshape((1024, ), input_shape=(32, 32, 1))(batch3)
        return Model(noise_input, result)

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

            # generate privatized images
            # prv_imgs = self.generator.predict(img_cons)
            prv_imgs = tf.add(self.generator.predict(tf.random.normal((4, 4, 256), self.mu, self.sigma)), imgs)

            # train the discriminator
            d_loss_prv = self.discriminator.train_on_batch(
                prv_imgs, gender_label)
            # print(d_loss_prv)

            # update penalty coefficient
            penalty_coef = epoch * 0.01

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(imgs, (imgs, gender_label))
            print("Epoch {0:3} [D loss: {1:20}, acc.: {2:8}%] [G loss: {3:20}]".format(
                epoch, d_loss_prv[0], 100*d_loss_prv[1], g_loss))


if __name__ == "__main__":
    gap = GAP()
    gap.train(epochs=20)

