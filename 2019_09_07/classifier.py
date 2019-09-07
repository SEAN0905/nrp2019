import numpy as np
import glob, os, random
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

base_path = "../dataset/face64"

# img_list = glob.glob(os.path.join(base_path, "*/*.jpg"))

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                # # remove unnecessary manipulation of the faces
                                #    shear_range=0.1,
                                #    zoom_range=0.1,
                                #    width_shift_range=0.1,
                                #    height_shift_range=0.1,
                                #    horizontal_flip=True,
                                #    vertical_flip=True,
                                   validation_split=0.2,
                                    )
                            

# test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.20)


# should create separate y_train instead
train_generator = train_datagen.flow_from_directory(base_path,
                                                    color_mode="grayscale",
                                                    target_size=(64, 64),
                                                    # 200 too big for batch size
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    shuffle=True)

# print(train_generator[0][0].shape)

validation_generator = train_datagen.flow_from_directory(base_path,
                                                         target_size=(64, 64),
                                                         color_mode="grayscale",
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         subset='validation',
                                                         shuffle=True)

# supposed to convert to a binary classification result
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

print(train_generator[0][1])


model = Sequential([
    Conv2D(
        filters=32,
        kernel_size=3,
        padding='same',
        activation='relu',
        # set reading mode to get "grayscale"
        input_shape=(64, 64, 1)),
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

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

model.summary()

callbacks = [
    EarlyStopping(monitor='val_acc', patience=200, verbose=1),
    ModelCheckpoint(filepath="model_checkpoint1.h5py",
                    monitor='val_acc',
                    save_best_only=True,
                    verbose=1)
]

history = model.fit_generator(train_generator,
                    # update based on change in batch size
                    steps_per_epoch=3571 // 32,
                    validation_steps=3571 // 32,
                    epochs=100,
                    validation_data=validation_generator,
                    callbacks=callbacks)

print(history.history.keys())
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("Classifier_accuracy.png")
plt.close()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("Classifier_loss.png")
plt.close()
