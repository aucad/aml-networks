# flake8: noqa: E402

"""
This script builds a neural network classifier for provided dataset.
Provide as input a path to a dataset, or script uses default
dataset if none provided. The dataset must be numeric
at all attributes.
"""

import warnings
from typing import Union, Optional

from art.estimators.classification import TensorFlowV2Classifier, \
    EnsembleClassifier, KerasClassifier, GPyGaussianProcessClassifier, \
    PyTorchClassifier, MXClassifier, TensorFlowClassifier
from art.experimental.estimators.classification import JaxClassifier

warnings.filterwarnings("ignore")

from art.estimators.classification.scikitlearn \
    import SklearnClassifier, ScikitlearnLogisticRegression, ScikitlearnSVC
from sklearn.neural_network import MLPClassifier
from art.defences.trainer import AdversarialTrainer, AdversarialTrainerMadryPGD
from art.attacks.evasion import FastGradientMethod

from src import Classifier

# AdversarialTrainer
NN_CLS_TYPE = Optional[Union[
    EnsembleClassifier, GPyGaussianProcessClassifier,
    KerasClassifier, JaxClassifier, MXClassifier, PyTorchClassifier,
    ScikitlearnLogisticRegression, ScikitlearnSVC, TensorFlowClassifier,
    TensorFlowV2Classifier]]

# AdversarialTrainerMadryPGD
# NN_CLS_TYPE = Optional[Union[
#     EnsembleClassifier, GPyGaussianProcessClassifier,
#     KerasClassifier, JaxClassifier, MXClassifier, PyTorchClassifier,
#     ScikitlearnLogisticRegression, ScikitlearnSVC, TensorFlowClassifier,
#     TensorFlowV2Classifier]]


class NeuralNetwork(Classifier):

    def __init__(self, *args):
        super().__init__('neural_network', *args)
        self.classifier: NN_CLS_TYPE = None

    @staticmethod
    def formatter(x, y):
        return x

    def predict(self, data):
        return self.model.predict(data)

    def init_weak(self):
        model = MLPClassifier()
        model.fit(self.train_x, self.train_y)
        return SklearnClassifier(model), model

    def init_robust(self, weak_cls):
        robust_model = MLPClassifier()
        robust_classifier = SklearnClassifier(robust_model)
        attack_fgm = FastGradientMethod(estimator=weak_cls, eps=0.15)
        trainer = AdversarialTrainer(
            classifier=robust_classifier,
            attacks=attack_fgm, ratio=0.5)
        trainer.fit(x=self.train_x, y=self.train_y, nb_epochs=10)
        self.classifier = trainer.get_classifier()
        self.model = robust_model

    def init_learner(self, robust):
        weak_cls, weak_model = self.init_weak()
        if robust:
            self.init_robust(weak_cls)
        else:
            self.classifier = weak_cls
            self.model = weak_model
