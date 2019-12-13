# to implement a GAP based on FNNP using OOP
# GAP: Generative Adversarial Privacy;
# FNNP: Feedforward neural network privatizer

# the dataset is face32_relabeled using text label

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization
from keras.layers import concatenate, Input, Reshape, LeakyReLU
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

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # to sepcify the input and output of the GAP
        z = Input(shape=(1124, ))
        img_prv = self.generator(z)

        # for the combined model only discriminator will be trained
        self.discriminator.trainable = False

        clasif_res = self.discriminator(img_prv)

        # TODO:to be tuned
        # the weight of the adversary loss
        self.loss_x = 0.1

        # the model yielf two results: img_prv and clasif_res
        # but with only one output
        self.combined = Model(z, clasif_res)
        self.combined.compile(optimizer=optimizer, loss=self.X_loss, loss_weights=[self.loss_x])

    def X_loss(self, y_true, y_pred):
        return K.mean(K.square(y_true - y_pred))

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

    def build_generator(self):
        # the function to initialize a privatizer(generator)
        # input: image concatenated with random noise vector
        # output: privatized image

        model = Sequential([
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

        model.summary()

        img_input = Input(shape=(1124, ))
        img_prv = model(img_input)

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
            gender_label =  Y_gender_raw[ids]
            smile_label = Y_smile_raw[ids]

            # generate random noise and concatenate to sampled images
            mu, sigma = 0, 0.1
            img_cons = np.asarray([
                np.append(np.reshape(X, (1024, 1)), np.random.normal(mu, sigma, 100))
                for X in imgs
            ])

            # generate privatized images
            prv_imgs = self.generator.predict(img_cons)

            # TODO: find an appropriate value for p
            #  set penalty coefficient
            p = 0.3

            # train the discriminator
            d_loss_prv = self.discriminator.train_on_batch(prv_imgs, gender_label)
            # print(d_loss_prv)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.combined.train_on_batch(img_cons, gender_label)
            print("Epoch {0:3} [D loss: {1:20}, acc.: {2:8}%] [G loss: {3:20}]".format(epoch, d_loss_prv[0], 100*d_loss_prv[1], g_loss))


if __name__ == "__main__":
    gap = GAP()
    gap.train(epochs=20)

            



