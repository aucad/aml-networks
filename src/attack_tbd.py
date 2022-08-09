"""
SOme black-box decision-based attack that works with XGBoost

TBD
"""

from sys import argv

# import numpy as np

import utility as tu


def run_attack(cls_loader, **cls_kwargs):
    classifier, model, attrs, x, y, _, _ = cls_loader(**cls_kwargs)

    # attack = ?
    # target_instance = x[0]
    # x_adv = attack.generate(x=np.array([target_instance]))
    # print(x_adv)
    print('done')


if __name__ == '__main__':
    from train_xg import train

    ds = argv[1] if len(argv) > 1 else tu.DEFAULT_DS
    run_attack(train, dataset=ds, test_size=0, max=-1, robust=False)
