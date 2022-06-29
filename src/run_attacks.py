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
from tree_xg import train_tree, formatter, predict
from attack_inf import inference_attack as inf
from attack_zoo import zoo_attack as zoo


def run_attacks(dataset):
    version = 'robust' if xgb.__version__ == '0.72' else 'non_robust'
    out_path = path.join('adversarial_xg', version)

    inf(train_tree,
        dataset=dataset, test_size=0.2)

    zoo(train_tree, formatter, predict, out_path,
        dataset=dataset, test_size=0)


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    run_attacks(ds)
