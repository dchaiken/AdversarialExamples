"""
Code to build the machine learning model
"""
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.core import Dropout


def get_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape, name="ipt"))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU())
    model.add(Dropout(.1))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU())
    model.add(Dropout(.1))
    model.add(MaxPool2D())
    # model.add(Conv2D(50, (3, 3)))
    # model.add(LeakyReLU())
    # model.add(Dropout(.1))
    # model.add(MaxPool2D())
    # model.add(Conv2D(50, (3, 3)))
    # model.add(LeakyReLU())
    # model.add(Dropout(.1))
    # model.add(Conv2D(25, (3, 3)))
    # model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(25))
    model.add(LeakyReLU())
    model.add(Dropout(.1))
    model.add(Dense(10, activation="softmax"))

    model.compile(optimizer="Nadam",
                  loss=categorical_crossentropy, metrics="accuracy")

    return model


if __name__ == "__main__":
    get_model((32, 32, 1))
