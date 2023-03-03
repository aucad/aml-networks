"""
Neural network classifier training.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
# from sklearn.neural_network import MLPClassifier

from src import Classifier


class NeuralNetwork(Classifier):

    def __init__(self, *args):
        super().__init__('neural_network', *args)

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def _set_cls(self, cls):
        self.classifier = cls
        self.model = cls.model

    def init_classifier(self):
        model = Sequential()
        model.add(Dense(
            units=100, activation='relu',
            kernel_regularizer=regularizers.L2(l2=1e-4), ))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss='binary_crossentropy', metrics=["accuracy"],
                      optimizer=Adam(learning_rate=0.001, clipvalue=None, ))
        model.fit(self.train_x, self.train_y,
                  shuffle=True, verbose=False, epochs=200, batch_size=200,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor='loss', patience=8, min_delta=1e-4)])
        return KerasClassifier(model=model, clip_values=(0, 1))

    def init_robust(self):
        robust_classifier = self.init_classifier()
        weak_classifier = self.init_classifier()
        attack_fgm = FastGradientMethod(weak_classifier)
        trainer = AdversarialTrainer(
            classifier=robust_classifier,
            attacks=attack_fgm, ratio=0.5)
        trainer.fit(
            self.train_x.copy(),
            self.train_y.copy(),
            nb_epochs=5, batch_size=128)
        self._set_cls(trainer.get_classifier())

    def init_learner(self, robust):
        if robust:
            self.init_robust()
        else:
            self._set_cls(self.init_classifier())
        # self.model = MLPClassifier()
        # self.model.fit(self.train_x, self.train_y)
        # self.classifier = SklearnClassifier(self.model)
