# to implement a GAP based on FNNP using OOP
# GAP: Generative Adversarial Privacy;
# FNNP: Feedforward neural network privatizer

# the dataset is face32_relabeled using text label

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import logging
tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel(logging.ERROR)

from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Activation
from keras.layers import concatenate, Input, Reshape, LeakyReLU, Lambda, Concatenate, GaussianNoise
from keras.activations import tanh
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD

dataset_path = "face32_relabeled/"


def pixel_mse_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def pixel_mae_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


class GAP():
    def __init__(self):
        self.optimizer = SGD(lr=0.09, momentum=0.9)

        # build generator and discrimninator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="categorical_crossentropy",
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])

        # to sepcify the input and output of the GAP
        z = Input(shape=(32, 32, 1))
        noise = Input(shape=(100, 1))
        img_prv = self.generator([z, noise])
        clasif_res = self.discriminator(img_prv)

        # the weight of the adversary loss, ie the penalty term
        self.loss_x = 2

        # the model takes two input: img_raw(z) and noise
        # yield two results: img_prv and clasif_res
        self.combined = Model([z, noise], [img_prv, clasif_res])

    def read_data(self):
        '''
        the function will return raw image data (scale to 1/255.)
        with gender and smile label in the format of [0, 1] or [1, 0]
        '''

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

    def build_generator(self):
        '''
        the function to build a privatizer(generator)
        # input: raw image
        # output: privatized image
        '''

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

    def build_discriminator(self):
        '''
        the function to initialize the pre-trained discriminator
        # input: privatized image from generator
        # output: classification result
        '''

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

    def train(self, epochs, mini_epochs, batch_size=64, sample_interval=50):
        '''
        the function will train the adversary model and combined model for given epoch number
        and will return the raw_imgs and privatized_imgs
        '''

        # load raw data
        X_data_raw, Y_gender_raw, Y_smile_raw = self.read_data()
        assert not np.any(np.isnan(X_data_raw))

        # print(X_data_raw.shape, Y_gender_raw.shape, Y_smile_raw.shape)
        # # (2723, 1024) (2723, 2) (2723, 2)

        # set baseline to update and save models of better performance
        pixel_mse_loss_min, epoch_min = 2, -1
        max_adversary, epoch_max = 0, -1

        for epoch in range(epochs):
            # select random batch of images and labels
            ids = np.random.randint(0, 2723-1, batch_size)
            imgs = X_data_raw[ids]
            # print(imgs.shape)
            gender_label = Y_gender_raw[ids]
            smile_label = Y_smile_raw[ids]

            # generate privatized images
            prv_imgs = self.generator.predict(
                [imgs, np.random.normal(0, 1, (batch_size, 100, 1))])
            # print(prv_imgs[0])

            # Train the Discriminator
            # train discriminator for a few epochs to better fit it into the privatized images
            for j in range(mini_epochs):
                d_loss_prv = self.discriminator.train_on_batch(
                    prv_imgs, gender_label)
                # print(d_loss_prv)
            
            # freeze discrimnator to train privatizer only
            self.discriminator.trainable = False

            # TODO: the loss weight for privatizer
            # update penalty coefficient and compile
            self.loss_x = epoch * 1 + 2

            self.combined.compile(optimizer=self.optimizer, loss=[
                                  pixel_mse_loss, "categorical_crossentropy"], loss_weights=[self.loss_x, -1])

            #  Train the Generator
            g_loss = self.combined.train_on_batch(
                [imgs, np.random.normal(0, 1, (batch_size, 100, 1))], [imgs, gender_label])
            
            # unfreeze the discriminator to make parameters saved 
            self.discriminator.trainable = True

            # separate line to make it easier to read
            print()
            print("loss_x: %.1f" % self.loss_x)
            print("Epoch %d [D loss: %.5f, acc. : %.3f %%] [G loss: combined: %.5f; pixel_mse_loss: %.7f; categorical_crossentropy: %.5f]" % (
                epoch, d_loss_prv[0], 100*d_loss_prv[1], g_loss[0], g_loss[1], g_loss[2]))
            print()

            # save the model based on its performance, return the output privatized images to display
            if g_loss[1] < pixel_mse_loss_min and g_loss[1] < 0.05 and d_loss_prv[1] < 0.6 and d_loss_prv[1] >= 0.4:
                # self.combined.save("GAN_FNNP_lr_weight_loss.h5")
                self.combined.save_weights("mse_save_weights.h5")
                pixel_mse_loss_min = g_loss[1]
                epoch_min = epoch
                return imgs, self.generator.predict([imgs, np.random.normal(0, 1, (batch_size, 100, 1))])
        # for easier track of the certain record
        print(epoch_min, pixel_mse_loss_min)


if __name__ == "__main__":
    gap = GAP()
    imgs, final_result = gap.train(epochs=50, mini_epochs=5)
    print(final_result)
    mini, maxi = 10, 0
    # 64 for batch size
    for i in range(64):
        o_img, p_img = imgs[i], final_result[i]
        # TODO: recover image from processed data
        img_name = str(k) + "_generate.png"
        img = Image.fromarray(np.uint8(p_img[k]), 'L')
        img.save(img_name)
        img_name = str(k) + "_original.png"
        img = Image.fromarray(np.uint8(o_img[k]), 'L')
        img.save(img_name)