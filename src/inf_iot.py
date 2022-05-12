"""
Simple adversarial example using ART with scikit-learn and applying
attribute inference attack.

Inspired by this example:

https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/
084e9769fc84934f77ec600ced5452a0e9baa60f/notebooks/
attack_attribute_inference.ipynb

Usage:

```
python src/inf_iot.py
```
"""

import warnings

warnings.filterwarnings("ignore")  # ignore import warnings

from art.estimators.classification.scikitlearn \
    import ScikitlearnDecisionTreeClassifier

from tree import train_tree
from utility import color_text as c


def attr_inference():
    # load model and data
    model, x_train, y_train, ATTR, x_test, y_test = \
        train_tree(False, test_set=0.2)

    art_classifier = ScikitlearnDecisionTreeClassifier(model)
    acc = model.score(x_test, y_test)
    print('Base model accuracy: ', c(f'{acc * 100:.2f} %'))
    print(c('done'))


if __name__ == '__main__':
    attr_inference()
