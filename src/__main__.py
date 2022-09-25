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
from os import path
from sys import exit
from typing import Optional, List

from attack_zoo import zoo_attack
from attack_hop import run_attack as hop_attack
from train_xg import train, formatter, predict
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

    if not dataset:
        parser.print_help()
        exit(1)
    else:
        __init_logger()
        run_attacks(dataset)


def plot_path(robust):
    img_base_path = 'robust' if robust else 'non_robust'
    return path.join(img_base_path)


def run_attacks(dataset):
    for opt in (NON_ROBUST, ROBUST):
        zoo_attack(
            train, formatter, predict,
            img_path=plot_path(opt),
            dataset=dataset,
            test_size=0,
            robust=opt)
        hop_attack(
            train, formatter, predict,
            dataset=dataset,
            test_size=0,
            robust=opt)


def __parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    # TODO: extend these args

    parser.add_argument(
        '-d', '--data',
        action="store",
        default=DEFAULT_DS,
        help="path to dataset",
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
