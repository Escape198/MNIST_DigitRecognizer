import tensorflow as tf
from data.download_data import load_data


def evaluate_model(model_path='mnist_model.h5'):
    _, (x_test, y_test) = load_data()
    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc
