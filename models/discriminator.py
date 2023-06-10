from constants import *

import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Input((HR_IMG_SIZE, HR_IMG_SIZE, 3)))

    model.add(layers.Conv2D(512, (3, 3), strides=(1, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (7, 7), strides=(2, 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
