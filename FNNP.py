# this file is to write FNNP based GAN using image data generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras import models, regularizers, backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, BatchNormalization, concatenate, Input, Reshape
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

base_path = "face32_gen/"

def load_data(dataset_type):
    dataset_path = base_path + "image_" + dataset_type
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        color_mode="grayscale",
        target_size=(32, 32),
        batch_size=32,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        color_mode="grayscale",
        target_size=(32, 32),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
    )
    print(train_generator[0][1])
    return train_generator, validation_generator


