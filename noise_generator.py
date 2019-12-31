import numpy as np
import math
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
from keras import models, regularizers, optimizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Activation, Add
from keras.layers import concatenate, Input, Reshape, LeakyReLU, Lambda, Concatenate, GaussianNoise
from keras.activations import tanh
from keras.losses import categorical_crossentropy
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD


def read_data():
    dataset_path = "face32_relabeled/"
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
        # for j, pixel_value in enumerate(data_raw):
        #     data_raw[j] = data_raw[j] / 255.0
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


def build_model():
    img_input = Input(shape=(32, 32, 1))
    noise_input = Input(shape=(32, 32, 1))
    result = Add()([img_input, noise_input])
    return Model([img_input, noise_input], result)


def build_discriminator():
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
    # model.load_weights("adversary32.h5")
    model.load_weights("smile_classifier.h5py")
    img_prv = Input(shape=(32, 32, 1))
    clasify_res = model(img_prv)
    return Model(img_prv, clasify_res)


mse_distortion = 0.0018308
actual_mse_distortion = mse_distortion * 256 * 256
sqrt_mse_distortion = math.sqrt(actual_mse_distortion)

# independent random noise generator
uniform_noise = np.random.uniform(
    low=0, high=mse_distortion*2, size=(2723, 32, 32, 1))

# independent random laplace noise generator
laplace_noise = np.random.laplace(
    loc=sqrt_mse_distortion, scale=mse_distortion, size=(2723, 32, 32, 1))

noise_trained_name = "uniform_noise"

if noise_trained_name == "uniform_noise":
    noise_training = uniform_noise
elif noise_trained_name == "laplace_noise":
    noise_training = laplace_noise

X_train_raw, Y_gender, Y_smile = read_data()
random_noise_model = build_model()
prv_imgs = random_noise_model.predict([X_train_raw, noise_training])
print("Noise model success")

prv_imgs_input = np.copy(prv_imgs)
for i in range(2723):
    for j in range(32):
        for k in range(32):
            prv_imgs_input[i][j][k] = prv_imgs_input[i][j][k] / 255.0

# gender_classifier = build_discriminator()
# gender_classifier.compile(optimizer=SGD(
#     lr=0.015, momentum=0.9), loss="categorical_crossentropy", metrics=["acc"])

smile_classifier = build_discriminator()
smile_classifier.compile(optimizer=SGD(
    lr=0.015, momentum=0.9), loss="categorical_crossentropy", metrics=["acc"])


print("Start to predict")
smile_acc = smile_classifier.evaluate(prv_imgs_input, Y_smile)
# smile_acc_or = smile_classifier.evaluate(X_train_raw, Y_smile)

print("Result:")
print("Noise training: ", noise_trained_name)
print("Noise parameter: mse: ", mse_distortion)
print("Smile accuracy: ", round(smile_acc[1], 3))
# print("Smile original accuracy: ", smile_acc_or)

# image_name = "test" + noise_trained_name + "_" + str(round(mse_distortion, 5))

# sample_raw_img = np.reshape(X_train_raw[111], (32, 32))
# # print(sample_raw_img)
# img = Image.fromarray(np.uint8(sample_raw_img), 'L')
# img.save(image_name + '_original.png')

# sample_prv_img = np.reshape(prv_imgs[111], (32, 32))
# img = Image.fromarray(np.uint8(sample_prv_img), 'L')
# img.save(image_name + '_prv.png')