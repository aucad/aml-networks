"""
Carry out the adversarial attacks using XGBoost classifier.

Usage:

```
python src/__main__.py
```

Use specific dataset:

```
python src/__main__.py ./path/to/input_data.csv
```

"""
import logging as lg
from argparse import ArgumentParser
from sys import exit
from typing import Optional, List

from loader import ClsLoader, AttackLoader
from utility import DEFAULT_DS

NON_ROBUST, ROBUST = False, True
VERSION = "0.1.0"
NAME = "src"


def main():
    """
    Run adversarial ML attacks and defenses on tree-based classifiers
    on network data.
    """
    parser = ArgumentParser(prog=NAME, description=main.__doc__)
    args = __parse_args(parser)
    dataset = args.data
    __init_logger()

    if not dataset:
        parser.print_help()
        exit(1)

    run_attacks(dataset, ClsLoader.XGBOOST)

    # TODO: apply selected args


def run_attacks(dataset, cls_kind=None):
    for opt in (NON_ROBUST, ROBUST):
        cls = ClsLoader.load(cls_kind).load(dataset, .98).train(robust=opt)
        AttackLoader.load(AttackLoader.HOP_SKIP).set_cls(cls).run()
        AttackLoader.load(AttackLoader.ZOO).set_cls(cls).run()


def __parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    parser.add_argument(
        '-d', '--data',
        action="store",
        default=DEFAULT_DS,
        help=f'Path to dataset [default: {DEFAULT_DS}]',
    )
    parser.add_argument(
        '-m', '--model',
        action='store',
        default=ClsLoader.XGBOOST,
        help=f'Learner. Options: {ClsLoader.DECISION_TREE} or '
             f'{ClsLoader.XGBOOST}. [default: {ClsLoader.XGBOOST}]'
    )
    parser.add_argument(
        "-r", "--robust",
        action='store_true',
        help="Use robust classifier (with defense)."
    )
    parser.add_argument(
        '-a', '--attack',
        action='store',
        help=f'Evasion attack. Options: '
             f'{AttackLoader.HOP_SKIP} or {AttackLoader.ZOO}'
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s " + VERSION,
    )
    return parser.parse_args(args)


def __init_logger(level: int = lg.ERROR, fn: Optional[str] = None):
    """Create a logger instance"""

    fmt = lg.Formatter(
        "[%(asctime)s] %(levelname)s (%(module)s): %(message)s",
        datefmt="%H:%M:%S")

    logger = lg.getLogger(NAME)
    logger.setLevel(level)
    stream_handler = lg.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if fn is not None:
        file_handler = lg.FileHandler(fn)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


if __name__ == '__main__':
    main()
