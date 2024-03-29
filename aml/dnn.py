"""
Neural network classifier implementation.

The classifier is built using Keras and conditionally with  robustness
from Adversarial Training.

paper: https://arxiv.org/abs/1705.07204 - adversarial training
paper: https://arxiv.org/abs/1607.02533 - BasicIterativeMethod
"""
from math import gcd

import tensorflow as tf

from keras import backend
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer

from aml import Classifier, utility

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class NeuralNetwork(Classifier):

    def __init__(self, *args):
        super().__init__('neural_network', *args)

    @staticmethod
    def formatter(x, y):
        return x

    def __c_key(self, key):
        return self.cls_conf[key] if key in self.cls_conf else {}

    def __model(self, key):
        m = self.__c_key('model')
        return m[key] if key in m else None

    def __m_train(self, key=None):
        m = self.__c_key('model_fit')
        return m if not key else (m[key] if key in m else None)

    def dnn_config(self):
        lrs = self.__model('layers') or \
              [60 for _ in range(max(1, len(self.mutable) // 4))]
        bs = gcd(self.n_train, (self.__m_train('batch_size') or 64))
        return lrs, {'epochs': 80, **self.__m_train(), 'batch_size': bs}

    def predict(self, data):
        tmp = self.model.predict(data)
        ax = 1 if len(tmp.shape) == 2 else 0
        return tmp.argmax(axis=ax)

    def _set_cls(self, cls):
        self.classifier = cls
        self.model = cls.model

    def init_classifier(self):
        """Trains a deep neural network classifier."""
        layers, args = self.dnn_config()
        layers = [Dense(v, activation='relu') for v in layers] + \
                 [Dense(self.n_classes, activation='softmax')]
        model = tf.keras.models.Sequential(layers)
        model.compile(
            optimizer=SGD(),
            loss=SparseCategoricalCrossentropy(),
            metrics=[SparseCategoricalAccuracy()])
        model.fit(
            self.train_x, self.train_y,
            shuffle=True, verbose=False, **args,
            callbacks=[EarlyStopping(monitor='loss', patience=5)])
        return KerasClassifier(model=model, clip_values=(0, 1))

    def init_robust(self):
        """Robust model training using Adversarial Training approach."""
        robust_classifier = self.init_classifier()
        attack = BasicIterativeMethod(
            robust_classifier, eps=0.3, eps_step=0.01, max_iter=40)
        trainer = AdversarialTrainer(
            # Model to train adversarially
            classifier=robust_classifier,
            # Attacks to use for data augmentation
            attacks=attack,
            # Proportion of samples to be replaced with adversarial
            # counterparts. Value 1 trains only on adversarial samples.
            ratio=0.5)
        trainer.fit(
            # Training set
            x=self.train_x.copy(),
            # Labels for the training set
            y=self.train_y.copy(),
            # Number of epochs to use for trainings
            nb_epochs=50, batch_size=50)
        utility.clear_one_line()
        self._set_cls(trainer.get_classifier())

    def init_learner(self, robust):
        self.init_robust() if robust \
            else self._set_cls(self.init_classifier())

    def cleanup(self):
        backend.clear_session()
