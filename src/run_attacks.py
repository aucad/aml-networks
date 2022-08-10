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

from attack_zoo import zoo_attack
from train_xg import train, formatter, predict
from utility import DEFAULT_DS

NON_ROBUST, ROBUST = False, True


def plot_path(robust):
    img_base_path = 'robust' if robust else 'non_robust'
    return path.join('results/xgboost', img_base_path)


def run_attacks(dataset):
    for opt in (NON_ROBUST, ROBUST):
        zoo_attack(
            train, formatter, predict,
            img_path=plot_path(opt),
            dataset=dataset,
            test_size=0,
            robust=opt)


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    run_attacks(ds)
