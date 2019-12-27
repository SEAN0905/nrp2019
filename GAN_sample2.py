from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers import Input, Reshape, Dense, Concatenate, LeakyReLU, Conv2D, BatchNormalization, MaxPool2D, Flatten
import time
from keras.utils import plot_model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import IPython.display as display
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
AUTOTUNE = tf.data.experimental.AUTOTUNE

##################################################################################################################
#     Define image directory, validation directory, Batch Size, Img Height, Img Width, epoch, optimizer
##################################################################################################################
print(tf.__version__)
data_dir = 'dataset'
val_dir = "validation"
data_dir = pathlib.Path(data_dir)
val_dir = pathlib.Path(val_dir)
image_count_train = len(list(data_dir.glob('*/*.jpg')))
image_count_val = len(list(val_dir.glob('*/*.jpg')))
total_train = image_count_train
BATCH_SIZE = 10
IMG_HEIGHT = 64
IMG_WIDTH = 64
epochs = 10
optimizer = Adam(0.005, 0.5)
STEPS_PER_EPOCH_TRAIN = np.ceil(image_count_train / BATCH_SIZE)
STEPS_PER_EPOCH_VAL = np.ceil(image_count_val / BATCH_SIZE)
CLASS_NAMES = np.array(
    [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES, image_count_train, image_count_val)

# Optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

################################################################
#                    Load Image
################################################################


def load_image(training_directory, validation_directory, batch_size, height, width):
    # load_image function is using image data generator to prepocessing the images and load the images in batch
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = ImageDataGenerator(rescale=1./255)
    # Generator for our validation data
    validation_image_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = image_generator.flow_from_directory(directory=str(training_directory),
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             target_size=(
                                                                 height, width),
                                                             class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(directory=str(validation_directory),
                                                                      batch_size=batch_size,
                                                                      shuffle=True,
                                                                      target_size=(
                                                                          height, width),
                                                                      class_mode='binary')
    print("Image Generator is built...\nTrain Data Generator is built...\nValidation Data Generator is built...\n")
    return train_data_gen, val_data_gen

################################################################
#                    Plot Images
################################################################


def plotImages(images_arr):
    """plotImages function for plotting figure"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

################################################################
#                    Privatizer
################################################################


def privatizer():
    """A privatizer function that take two inputs: img and noise to generate a new privatized image"""
    img = Input(shape=(64, 64, 3))
    x = Reshape(target_shape=(1, 12288))((img))
    # assert x.output_shape == (None, 1, 12288) # Note: None is the batch size
    noise = Input(shape=(100,))
    y = Reshape((1, 100))(noise)
    #assert y.shape == (None, 1, 100)
    x = Concatenate()([x, y])
    #assert x.output_shape == (None, 1, 12388)

    x = Dense(12288)(x)
    x = LeakyReLU()(x)
    #assert x.output_shape == (None, 1, 12288)
    x = Dense(12288)(x)
    x = LeakyReLU()(x)
    #assert x.output_shape == (None, 1, 12288)
    x = Dense(12288)(x)
    x = LeakyReLU()(x)
    #assert x.output_shape == (None, 1, 12288)
    x = Dense(12288)(x)
    x = LeakyReLU()(x)
    #assert x.output_shape == (None, 1, 12288)
    img_gen = Reshape((64, 64, 3))(x)
    #assert x.output_shape == (None, 64, 64, 3)
    return Model(inputs=[img, noise], outputs=img_gen)

################################################################
#                    Discriminator
################################################################


def discriminator():
    """A discriminator function to disciminate the generated image"""
    img = Input(shape=(64, 64, 3))
    x = Conv2D(filters=32,
                    kernel_size=(3, 3),
                    input_shape=(64, 64, 3),
                    activation="relu")(img)

    x = BatchNormalization()(x)
    x = MaxPool2D()(x)

    x = Conv2D(filters=64,
                    kernel_size=(3, 3),
                    activation="relu")(x)

    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)

    x = Dense(units=1024,
                    activation="relu")(x)

    x = Dense(units=1024,
                    activation="relu")(x)

    validity = Dense(units=2,
                            activation='softmax')(x)

    return Model(inputs=img, outputs=validity)


def discriminator_loss(real_output, fake_output):

    """ A discriminator_loss function to compute the total loss"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(y_true, y_pred):
    """A generator_loss function to compute the pixel loss of img with generated img"""
    return K.mean(K.square(y_true - y_pred)) / (255.0 * 255.0)

################################################################
#     Get image data generator and plot the first 5 images
################################################################
train_data_gen, val_data_gen = load_image(data_dir, val_dir, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH)
#sample_training_images, _ = next(train_data_gen)
# plotImages(sample_training_images[:5])

################################################################
#               Build privatizer and discriminator
################################################################

privatizer = privatizer()
discriminator = discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

################################################################
#                           Train
################################################################
# Adversarial ground truths
valid = np.ones((BATCH_SIZE, 2))
fake = np.zeros((BATCH_SIZE, 2))
for i in range(3):
    X_train, _ = next(train_data_gen)
plotImages(X_train[:5])

print("X_train's shape" + str(X_train.shape))
gen_imgs = privatizer.predict([X_train, np.random.normal(0, 1, (BATCH_SIZE, 100))])
print("Generated image's shape" + str(gen_imgs.shape))
# Rescale it back to 0 ~255 then plot the generated images of Generator
plotImages((gen_imgs[:5] * 255).astype(np.uint8))
validity = discriminator(privatizer.output)
print(validity.shape)
discriminator.trainable = False

# accepting two input for model privitizer and discriminator
combined = Model(privatizer.input, [validity, privatizer.output])
combined.compile(loss=generator_loss, optimizer=optimizer)
# Train the discriminator
d_loss_real = discriminator.train_on_batch(X_train, valid)
# K.constant turn gen_imgs into tensor
d_loss_fake = discriminator.train_on_batch(K.constant(gen_imgs), fake)
d_loss = discriminator_loss(d_loss_real, d_loss_fake)

# --------------------
#  Train Generator
# --------------------

# Train the generator (to have the discriminator label samples as valid)
g_loss = combined.train_on_batch([X_train, np.random.normal(0, 1, (BATCH_SIZE, 100))], valid)
#print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
