"""
Code to load in train, val, and test datasts
"""
from keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical


def load_images():
    """
    Load MNIST data (using MNIST because of both ease and visibility of perturbations)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    num_training_samples = int(X_train.shape[0]*5/6)
    X_val = X_train[num_training_samples:]
    X_train = X_train[:num_training_samples]
    y_val = y_train[num_training_samples:]
    y_train = y_train[:num_training_samples]
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    print(load_images()[0][0].shape)
