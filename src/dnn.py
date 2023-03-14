"""
Neural network classifier implementation.

The classifier is built using Keras and conditionally with  robustness
from Adversarial Training.

paper: https://arxiv.org/abs/1705.07204 (adversarial training)
"""

import os
import warnings


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras import backend as K
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer

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

    def init_classifier(self, epochs=180, batch_size=256):
        model = tf.keras.models.Sequential([
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(100, activation='relu'),
            Dense(self.n_classes, activation='softmax')
        ])
        model.compile(
            optimizer=SGD(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()]
        )
        model.fit(
            self.train_x, self.train_y,
            batch_size=min(batch_size, self.n_train),
            callbacks=[EarlyStopping(monitor='loss', patience=5)],
            epochs=epochs,
            shuffle=True,
            verbose=False,
        )
        return KerasClassifier(model=model)

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
            nb_epochs=5,
            batch_size=min(128, self.n_train))
        self._set_cls(trainer.get_classifier())

    def init_learner(self, robust):
        if robust:
            self.init_robust()
        else:
            self._set_cls(self.init_classifier())

    def cleanup(self):
        K.clear_session()
