import logging

logger = logging.getLogger(__name__)


class AbsClassifierInstance:

    def __init__(self):
        self.classifier = None
        self.model = None
        self.attrs = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

    @staticmethod
    def formatter(x, y):
        return x

    @staticmethod
    def predict(model, data):
        return model.predict(data)

    def plot(self):
        pass

    def train(self):
        pass
