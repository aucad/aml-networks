"""
Carry out the adversarial attacks using XGBoost classifier.

Usage:

```
python src/run_attacks.py
```

Use specific dataset:

```
python src/run_attacks.py ./path/to/input_data.csv
```
"""
from os import path
from sys import argv

import xgboost as xgb

from tree_utils import DEFAULT_DS
from tree_xg import train_boosted_tree
from attack_inf import inference_attack
from attack_zoo import zoo_attack


def run_attacks(dataset):
    inference_attack(
        train_boosted_tree, dataset=dataset, test_size=0.2)

    zoo_attack(
        train_boosted_tree, path.join('boosted', 'non_robust'),
        dt_formatter=lambda x: xgb.DMatrix(x),
        dataset=dataset, test_size=0)


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    run_attacks(ds)
