"""
Create adversarial examples from the MNIST Testing data set.

Ended up referencing:
https://stackoverflow.com/questions/59145221/how-to-compute-gradient-of-output-wrt-input-in-tensorflow-2-0
https://github.com/tensorflow/tensorflow/issues/30190
http://blog.ai.ovgu.de/posts/jens/2019/001_tf20_pitfalls/index.html

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_processing import load_images
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow import GradientTape

loss_object = tf.keras.losses.CategoricalCrossentropy()


def get_loss(model, X, y_true):
    y_pred = model(X)
    print(y_pred)
    print(y_true)
    return loss_object(y_true=y_true, y_pred=y_pred)


model = load_model("latest_model")

X_train, y_train, X_val, y_val, X_test, y_test = load_images()
correct_count = 0
misled_count = 0

# Gradients are multiplied by coeff over mean absolute value of the gradients
coeff = 3.5

for ind in range(1000):
    inputs = X_test[ind][np.newaxis, ...]
    inputs = tf.Variable(inputs, dtype=tf.float64)
    truth = tf.Variable(y_test[ind][np.newaxis, ...], dtype=tf.float64)

    with GradientTape(persistent=True) as tape:
        y_pred = model(inputs)
        loss_value = loss_object(y_true=truth, y_pred=y_pred)
        # print(loss_value)
    # Possible to compute higher-order derivates, do approximation analogous to Runge-Kutta?
    grads = tape.gradient(loss_value, inputs)
    meanval = np.mean(np.abs(grads))
    # print(grads)

    # Create a new image by moving in the direction of the gradients
    edited = inputs + tf.Variable(coeff/meanval, dtype=tf.float64) * grads
    y_true_val = np.argmax(y_test[ind])
    y_pred_val = np.argmax(y_pred[0])
    # Only care about misleading the model, so ignore initially misclassified examples
    if y_pred_val == y_true_val:
        correct_count += 1
        # Some examples have vanishing gradient that approximates to 0 by the input layer
        if meanval != 0:
            y_pred_edited_val = np.argmax(model(edited)[0])
            print(ind, y_true_val, y_pred_val, y_pred_edited_val)
            if y_pred_edited_val != y_true_val:
                im = array_to_img(inputs[0])
                im.save("test_img.png")
                grads_im = array_to_img(grads.numpy()[0])
                grads_im.save("grads_img.png")
                edited_im = array_to_img(edited.numpy()[0])
                edited_im.save("edited_img.png")
                misled_count += 1

print(correct_count)
print(misled_count)
