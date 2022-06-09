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

from tree_utils import DEFAULT_DS
from tree_xg import train_tree, formatter, predict
from attack_inf import inference_attack as inf
from attack_zoo import zoo_attack as zoo


def run_attacks(dataset):
    out_path = path.join('boosted', 'non_robust')

    inf(train_tree,
        dataset=dataset, test_size=0.2)

    zoo(train_tree, formatter, predict, out_path,
        dataset=dataset, test_size=0)


if __name__ == '__main__':
    ds = argv[1] if len(argv) > 1 else DEFAULT_DS
    run_attacks(ds)
