# the privatizer model based on FNNP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout, concatenate, Input, Reshape
from keras.models import Sequential, Model, load_model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import backend as K

base_path = "face32_relabeled/"

# def load_data():
#     train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
#     train_generator = train_datagen.flow_from_directory(
#         base_path,
#         color_mode="grayscale",
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode="categorical",
#         subset="training",
#         shuffle=True)
#     validation_generator = train_datagen.flow_from_directory(
#         base_path,
#         color_mode="grayscale",
#         target_size=(64, 64),
#         batch_size=32,
#         class_mode="categorical",
#         subset="validation",
#         shuffle=True)
#     print(train_generator[0][1])
#     return train_generator, validation_generator


def read_data():
    handle1 = open(base_path + "gender.txt", "r")
    handle2 = open(base_path + "smile.txt", "r")

    raw_gender_label = []
    for line in handle1:
        line = line.strip()
        if line == "0" or line == "1":
            raw_gender_label.append(line)
        else:
            print(line)
    handle1.close()
    # # print(raw_gender_label[1230])
    # print("r", len(raw_gender_label))
    # # 3986

    raw_smile_label = []
    for line in handle2:
        line = line.strip()
        if line == "0" or line == "1":
            raw_smile_label.append(line)
        else:
            print(line)
    handle2.close()
    # print(raw_smile_label[1230])
    # print(len(raw_smile_label))

    x_train, Y_gender, Y_smile = [], [], []
    # supposed to be 1164 in the end, merely a counter for check
    broken_series = 0
    for i in range(1, 2723 + 1):
        image_name = base_path + "image/" + str(i) + ".jpg"
        try:
            image = Image.open(image_name)
            image.load()
        except:
            # print(image_name)
            broken_series += 1
            continue
        image_gender_label = raw_gender_label[i - 1]
        image_smile_label = raw_smile_label[i - 1]
        # print(image_gender_label, image_smile_label)
        raw_data = np.asarray(image, dtype="int32")
        data = np.reshape(raw_data, (1024, ))
        x_train.append(data)
        Y_gender.append([image_gender_label == 'man', image_gender_label == 'woman'])
        Y_smile.append([image_smile_label == '1', image_smile_label == '0'])
    print(Y_gender[1:6])
    print(Y_smile[1:6])
    print("b", broken_series)
    return np.asarray(x_train), np.asarray(Y_gender), np.asarray(Y_smile)


# TODO: add in the validation and test set
# train, validate = load_data()
X_data, Y_gender, Y_smile = read_data()

# TODO: mu, sigma value to be decided
mu, sigma = 0, 0.1
# noise = np.random.normal(mu, sigma, 100)

X_data_raw = np.copy(X_data)

X_data = np.asarray([
    np.append(np.reshape(X, (1024, 1)), np.random.normal(mu, sigma, 100))
    for X in X_data
])

validation_split = 0.1
train_num = int(len(X_data) * (1 - validation_split))

X_train = X_data[:train_num]
X_test = X_data[train_num:]
# # could use the line below for conversion
# y_gender = np_utils.to_categorical(y_gender, 2)
y_gender_train = Y_gender[:train_num]
y_gender_test = Y_gender[train_num:]
print(y_gender_test.shape)
y_smile_train = Y_smile[:train_num]
y_smile_test = Y_smile[train_num:]

X_train_raw = X_data_raw[:train_num]
X_test_raw = X_data_raw[train_num:]

print(X_train.shape)

privatizer = Sequential([
    Dense(1124, input_shape=(1124,)),
    keras.layers.LeakyReLU(alpha=0.3),
    Dense(1024),
    keras.layers.LeakyReLU(alpha=0.3),
    Dense(1024),
    keras.layers.LeakyReLU(alpha=0.3),
    Dense(1024),
    keras.layers.LeakyReLU(alpha=0.3),
    Reshape((32, 32, 1)),
])
# privatizer.compile(loss="categorical_crossentropy", optimizer="sgd")

privatizer.inputs = X_train

# supposed to load the save weights of FNNP_model
# privatizer = keras.models.load_model("iufgefyqj.h5")
for item in privatizer.layers:
    item.trainable = False

# # TODO: to be updated
# privatizer.compile(optimizer=SGD(lr=0.1, momentum=0.9),
#                    loss=["categorical_crossentropy"])
# privatizer.summary()

# assume adversary model is saved as a.h5

# gender_classifier = load_model("random.h5")

gender_classifier = Sequential([
    Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        # set reading mode to get "grayscale"
        input_shape=(32, 32, 1)),
    BatchNormalization(),
    # # remove additional layer coming from imagination
    # Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dense(1024, kernel_initializer='random_uniform', activation='relu'),
    BatchNormalization(),
    Dense(1024, kernel_initializer='random_uniform', activation='relu'),
    BatchNormalization(),
    Flatten(),
    # supposed to be 2 for prediction, one predicting smile and the other not
    Dense(2, activation='softmax')
])

gender_classifier.trainable = False
# TODO: to update the optimizer and loss
gender_classifier.compile(optimizer=SGD(lr=0.1, momentum=0.9),
                          loss=["categorical_crossentropy"],
                          metrics=['accuracy'])

overall_model = Model(
    input=privatizer.input,
    output=[gender_classifier(privatizer.output), privatizer.layers[-2].output])
overall_model.summary()


def X_loss(y_true, y_predicted):
    return K.mean(K.square(y_true - y_predicted) / (255.0 * 255.0))


n_epoch = 30
lr = 0.002

N_layer_p = len(privatizer.layers)

num_iter = 1
while True:
    # loss_x = float(input("Input the penalty parameter"))
    loss_x = 1

    # train the privatizer model and set adversary to be untrainable
    for item in overall_model.layers[:N_layer_p]:
        item.trainable = True
    overall_model.layers[-1].trainable = False
    print(X_test.shape)
    overall_model.compile(optimizer=SGD(lr=lr, momentum=0.90),
                          loss=["categorical_crossentropy", X_loss],
                          loss_weights=[1, loss_x],
                          )

    overall_model.summary()

    overall_model.fit(x=X_train,
                      y=[y_gender_train, X_train_raw],
                      batch_size=60,
                      epochs=n_epoch,
                      validation_data=([X_test], [y_gender_test, X_test_raw]))
    
    # opposite as block above
    for item in overall_model.layers[:N_layer_p]:
        item.trainable = False
    overall_model.layers[-1].trainable = True

    overall_model.compile(optimizer=SGD(lr=lr, momentum=0.90),
                          loss=["categorical_crossentropy", X_loss],
                          loss_weights=[1, loss_x],
                          metrics=[["accuracy"], [X_loss]])
    
    overall_model.fit(x=X_train,
                      y=[y_gender_train, X_train_raw],
                      batch_size=60,
                      epochs=n_epoch,
                      validation_data=([X_test], [y_gender_test, X_test_raw]))

    num_iter -= 1
    if num_iter < 0:
        break
    # overall_model

evaluate_metric = overall_model.evaluate(x=X_data, y=[Y_smile, Y_gender])
print(evaluate_metric)
overall_model.save("GAN.h5")