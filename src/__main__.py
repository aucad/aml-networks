"""
Command-line interface for training tree-based classifiers and
performing adversarial attacks on those models.

Usage:

```
python -m src
```

List all available options:

```
python -m src --help
```

"""
import logging
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from sys import exit
from typing import Optional, List

from src import __version__, __title__, Experiment


def main():
    """
    Run adversarial ML attacks and defenses on tree-based classifiers
    on network data.
    """
    parser = ArgumentParser(prog=__title__, description=main.__doc__)
    args = __parse_args(parser)
    save_log = not args.no_log

    if not args.dataset:
        parser.print_help()
        exit(1)

    ts = str(round(time.time() * 1000))[-4:]
    ln = log_name(ts, args)
    ensure_out_dir(args.out)

    __init_logger(
        level=logging.FATAL - (0 if args.silent else 40),
        fn=ln if save_log else None)

    Experiment(ts, **args.__dict__).run()

    if save_log:
        print('Log file:', ln)


def __init_logger(level: int, fn: str = None):
    """Create a logger instance"""
    fmt = logging.Formatter("%(message)s")

    logger = logging.getLogger(__title__)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if fn is not None:
        file_handler = logging.FileHandler(filename=f'{fn}_log.txt')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def ensure_out_dir(dir_path):
    return os.path.exists(dir_path) or os.makedirs(dir_path)


def log_name(ts, args):
    ds_name = Path(args.dataset).stem
    token = f'robust{"T" if args.robust else "F"}_{ds_name}_{args.attack}'
    return os.path.join(args.out, f'{token}_{ts}')


def __parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    parser.add_argument(
        '-d', '--dataset',
        action="store",
        default=Experiment.DEFAULT_DS,
        help=f'path to dataset [default: {Experiment.DEFAULT_DS}]',
    )
    parser.add_argument(
        '-k', '--kfolds',
        type=int,
        choices=range(1, 11),
        metavar="1-10",
        help='K-folds number of splits [default: 5]',
        default=5
    )
    parser.add_argument(
        '-i', '--iter',
        type=int,
        choices=range(1, 10000),
        metavar="1-10000",
        help='max attack iterations [default: not set]',
        default=0
    )
    parser.add_argument(
        '-c', '--cls',
        action='store',
        choices=Experiment.CLASSIFIERS,
        default=Experiment.DEFAULT_CLS,
        help=f'Classifier to train [default: {Experiment.DEFAULT_CLS}]'
    )
    parser.add_argument(
        "--robust",
        action='store_true',
        help="train a robust model (xgboost only)"
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        help="generate plots"
    )
    parser.add_argument(
        '-a', '--attack',
        action='store',
        choices=Experiment.ATTACKS,
        help=f'evasion attack [default: None]'
    )
    parser.add_argument(
        '--validator',
        action='store',
        choices=Experiment.VALIDATORS,
        help=f'dataset validator kind [default: None]'
    )
    parser.add_argument(
        "-o", "--out",
        action='store',
        default="output",
        help="output directory [default: output]"
    )
    parser.add_argument(
        "--no_log",
        action='store_true',
        help="do not save terminal output to a file",
    )
    parser.add_argument(
        "--save_rec",
        action='store_true',
        help="save records (original and adversarial)",
    )
    parser.add_argument(
        '-s', "--silent",
        action='store_true',
        help="disable debug logging"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
