import tensorflow as tf
from data.download_data import load_data
from models.build_model import build_model


def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=5)
    model.save('mnist_model.h5')
    return model
