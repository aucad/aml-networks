# flake8: noqa: E402

"""
This script builds a neural network classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.
"""

import warnings

from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

from typing import Union, Optional

from art.estimators.classification import TensorFlowV2Classifier, \
    EnsembleClassifier, KerasClassifier, GPyGaussianProcessClassifier, \
    PyTorchClassifier, MXClassifier, TensorFlowClassifier
from art.experimental.estimators.classification import JaxClassifier


from art.estimators.classification.scikitlearn import \
    ScikitlearnLogisticRegression, ScikitlearnSVC, ScikitlearnClassifier
from art.defences.trainer import AdversarialTrainer, \
    AdversarialTrainerMadryPGD
from art.attacks.evasion import FastGradientMethod

from src import Classifier

# AdversarialTrainer, AdversarialTrainerMadryPGD
NN_CLS_TYPE = Optional[Union[
    EnsembleClassifier, GPyGaussianProcessClassifier,
    KerasClassifier, JaxClassifier, MXClassifier, PyTorchClassifier,
    ScikitlearnLogisticRegression, ScikitlearnSVC, TensorFlowClassifier,
    TensorFlowV2Classifier]]


class NeuralNetwork(Classifier):

    def __init__(self, *args):
        super().__init__('neural_network', *args)
        self.classifier: NN_CLS_TYPE = None

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    @staticmethod
    def init_model():
        return MLPClassifier()

    @staticmethod
    def init_classifier(model) -> NN_CLS_TYPE:
        # FIXME: return compatible type
        return ScikitlearnClassifier(model)

    def init_weak(self):
        model = self.init_model()
        model.fit(self.train_x, self.train_y)
        return self.init_classifier(model), model

    def init_robust(self):
        # TODO
        self.model = self.init_model()
        robust_classifier = self.init_classifier(self.model)
        weak_classifier, _ = self.init_weak()
        attack_fgm = FastGradientMethod(weak_classifier)
        trainer = AdversarialTrainer(
            classifier=robust_classifier,
            attacks=attack_fgm, ratio=0.5)
        trainer.fit(self.train_x.copy(),
                    self.train_y.copy(),
                    nb_epochs=5, batch_size=128)
        self.classifier = trainer.get_classifier()

    def init_learner(self, robust):
        # if robust:
        #     self.init_robust()
        # else:
        self.classifier, self.model = self.init_weak()
