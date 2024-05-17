from training.train_model import train_model
from evaluation.evaluate_model import evaluate_model


if __name__ == '__main__':
    model = train_model()
    evaluate_model()
