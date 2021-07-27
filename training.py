"""
Basic code to train a model.

Future work might include training using adversarial examples,
as well as preprocessing training data to include random noise/rotations,
and evaluating the difficulty of generating adversarial examples afterwards
"""

from data_processing import load_images
from model import get_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_images()
    model = get_model((28, 28, 1))
    model.fit(X_train, y_train, validation_data=(
        X_val, y_val), epochs=20, batch_size=64, verbose=1)
    model.save("latest_model")
    loss, accuracy = model.evaluate(X_val, y_val, batch_size=64)

    print("accuracy: {}%".format(accuracy*100))
