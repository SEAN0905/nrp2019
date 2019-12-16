# to implement a GAP based on FNNP using OOP
# GAP: Generative Adversarial Privacy;
# FNNP: Feedforward neural network privatizer

# the dataset is face32_relabeled using text label

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
from keras.layers import concatenate, Input, Reshape, LeakyReLU, Lambda, Concatenate, GaussianNoise
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam

dataset_path = "face32_relabeled/"


class GAP():
    def __init__(self):
        # set image input basic information
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = SGD(lr=0.01, momentum=0.9)

        # build generator
        self.generator = self.build_generator()

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="categorical_crossentropy",
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # for the combined model only discriminator will be trained
        self.discriminator.trainable = False

        # to sepcify the input and output of the GAP
        z = Input(shape=(1024, ))
        img_prv = self.generator(z)

        clasif_res = self.discriminator(img_prv)

        # TODO:to be tuned
        # the weight of the adversary loss
        self.loss_x = 0.1

        # set penalty term for privatizer loss function
        self.penalty_coef = 0.01

        # distortion to limit change to original image
        self.distortion = 0.1

        # the model yield two results: img_prv and clasif_res
        self.combined = Model(z, (img_prv, clasif_res))
        self.combined.compile(optimizer=optimizer, loss=[
                              self.privatizer_loss, "categorical_loss"], loss_weights=[1, self.loss_x])

    def privatizer_loss(self, y_true, y_pred, eps=1e-15):
        # Prepare numpy array data
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        assert (len(y_true) and len(y_true) == len(y_pred))

        # Clip y_pred between eps and 1-eps
        p = np.clip(y_pred, eps, 1-eps)
        loss = np.sum(- y_true * np.log(p) - (1 - y_true) * np.log(1-p))

        log_loss = loss / len(y_true)

        # penalty term
        # TODO: CHECK the loss function in paper suggests the distortion to x rather than y
        # so here, instead of y_true, y_pred, should be x, y_pred?
        # but how to show that then?
        distortion_punishment = self.penalty_coef * \
            max(0, np.mean((np.square(y_true - y_pred) - self.distortion)))

        return log_loss + distortion_punishment

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
            data = np.reshape(raw_image, (1024, ))
            X_train.append(data)
            Y_gender.append([image_gender_label == "0",
                             image_gender_label == "1"])
            Y_smile.append([image_smile_label == "0",
                            image_smile_label == "1"])
        # print(Y_gender[1:6])
        # print(Y_smile[1:6])
        # # [[False, True], [False, True], [True, False], [True, False], [True, False]]
        # # [[False, True], [False, True], [False, True], [False, True], [False, True]]
        return np.asarray(X_train), np.asarray(Y_gender), np.asarray(Y_smile)

    # def concatenate(self, X_raw):
    #     # print(X_raw.shape)

    #     # set parameter for random noise vector
    #     mu, sigma = 0, 0.1

    #     random_noise = np.random.normal(mu, sigma, (X_raw.shape[0], 100))

    #     print(X_raw.shape)
    #     print(random_noise.shape)
    #     # X_processed = tf.map_fn(fn=lambda x:np.append(x, random_noise), elems=X_raw)

    #     # return Model(X_raw, X_processed)
    #     return K.concatenate([X_raw, random_noise], axis=1)

    def build_generator(self):
        # the function to initialize a privatizer(generator)
        # input: image concatenated with random noise vector
        # output: privatized image

        mu, sigma = 0, 0.1

        # random_noise = tf.random.normal((100, 1), mu, sigma)
        
        # def img_cat(x_raw):
        #     return tf.map_fn(lambda x: K.concatenate(
        #         [x, tf.random.normal((100, 1), mu, sigma)], axis=0), x_raw)

        # def _cat(x_raw):
        #     return np.asarray(np.append(x, np.random.normal(mu, sigma, (100, 1))) for x in x_raw)

        # model = Sequential([
        #     # Lambda(img_cat, input_shape=(1024, 1)),
        #     # Concatenate([Input(shape=(1024, 1)), np.random.normal(mu, sigma, (100, 1))], input_shape=(1024, 1)),
        #     Dense(1, input_shape=(1124, 1)),
        #     Dense(1124),
        #     LeakyReLU(alpha=0.3),
        #     Dense(1024),
        #     LeakyReLU(alpha=0.3),
        #     Dense(1024),
        #     LeakyReLU(alpha=0.3),
        #     Dense(1024, activation=LeakyReLU(alpha=0.3)),
        #     Dense(1),
        #     Reshape((32, 32, 1), input_shape=(1024, 1))
        # ])

        # model.summary()

        img_input = Input(shape=(1024, 1))
        img_input_dense = Dense(1, activation="relu")(img_input)
        '''
        the following part of the function contains the bug regarding use of tensor
        the code aims to concatenate the random noise to the image input
        '''

        img_cat = tf.map_fn(lambda x: K.concatenate(
            [x, tf.random.normal((100, 1), mu, sigma)], axis=0), img_input)
        
        noise = tf.random.normal((100, 1), mu, sigma)
        noise_dense = Dense(1, )(noise)

        # noise_dense = GaussianNoise(sigma, input_shape=(100, ))


        # img_cat = Concatenate()([img_input_dense, noise_dense])

        # img_cat = tf.map_fn(lambda x: K.concatenate([x, noise_dense], axis=0), img_input_dense)
        img_cat = Lambda(tf.map_fn(lambda x: K.concatenate([x, noise_dense], axis=0), img_input_dense), output_shape=(1124, 1))(img_input_dense)
        # model.add(Lambda(lambda x: max(0., min(x,100.)), output_shape=(1,)))

        # print(img_cat.shape)
        # # (?, 1124, 1)

        receiver = Flatten()(img_cat)
        dense1 = Dense(1024, activation=LeakyReLU())(receiver)
        dense2 = Dense(1024, activation=LeakyReLU())(dense1)
        dense3 = Dense(1024, activation=LeakyReLU())(dense2)
        dense4 = Dense(1024, activation=LeakyReLU())(dense3)
        print("dense4: ", dense4)
        img_prv = Reshape((32, 32, 1), input_shape=(1024, ))(dense4)
        print("img_prv", img_prv)


        # img_cat = Lambda(_cat, output_shape=(1124, 1)).output
        # img_cat = Concatenate([img_input, tf.random.normal((100, 1), mu, sigma)], input_shape=(1024, 1))
        #  img_cat = model2(img_input)
        # img_raw = tf.map_fn(lambda x: K.concatenate(
        #     [x, tf.random.normal((100, ), mu, sigma)]), img_input)

        # img_prv = K.reshape(model(img_input), (32, 32, 1))
        # img_prv = model(img_cat)
        # img_prv_rshp = K.reshape(K.expand_dims(img_prv), (32, 32, 1))

        return Model(img_input, img_prv)

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

            # # generate random noise and concatenate to sampled images
            # mu, sigma = 0, 0.1
            # img_cons = np.asarray([
            #     np.append(np.reshape(X, (1024, 1)), np.random.normal(mu, sigma, 100))
            #     for X in imgs
            # ])

            # generate privatized images
            # prv_imgs = self.generator.predict(img_cons)
            prv_imgs = self.generator.predict(imgs)

            # train the discriminator
            d_loss_prv = self.discriminator.train_on_batch(
                prv_imgs, gender_label)
            # print(d_loss_prv)

            # update penalty coefficient
            self.penalty_coef = epoch * 0.01

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(imgs, (imgs, gender_label))
            print("Epoch {0:3} [D loss: {1:20}, acc.: {2:8}%] [G loss: {3:20}]".format(
                epoch, d_loss_prv[0], 100*d_loss_prv[1], g_loss))


if __name__ == "__main__":
    gap = GAP()
    gap.train(epochs=20)
