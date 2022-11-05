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
from argparse import ArgumentParser
from pathlib import Path
from sys import exit
from typing import Optional, List

from src import __version__, __title__, Experiment, Validator, utility


def main():
    """
    Run adversarial ML attacks and defenses on tree-based classifiers
    on network data.
    """
    parser = ArgumentParser(prog=__title__, description=main.__doc__)
    args = parse_args(parser)

    if not args.dataset:
        parser.print_help()
        exit(1)

    is_exp, is_validator = args.which == 'exp', args.which == 'vld'
    ts = utility.ts_str()
    utility.ensure_dir(args.out)
    save_log = is_exp and not args.no_log and not args.silent
    ln = log_name(ts, args) if save_log else None
    log_level = logging.FATAL if args.silent else logging.DEBUG
    init_logger(log_level, ln)

    if is_validator:
        Validator.validate_dataset(
            args.validator, args.dataset, args.capture, args.out)

    if is_exp:
        Experiment(ts, **args.__dict__).run()

    if ln:
        print('Log file:', ln)


def init_logger(level: int, fn: str = None):
    """Create a logger instance"""
    fmt = logging.Formatter("%(message)s")

    logger = logging.getLogger(__title__)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if fn is not None:
        file_handler = logging.FileHandler(filename=f'{fn}')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def log_name(ts, args):
    ds = Path(args.dataset).stem
    rb = f'robust_{"T" if args.robust else "F"}_' \
        if hasattr(args, 'robust') else ''
    atk = f'{args.attack}_' if hasattr(args, 'attack') else ''
    return os.path.join(args.out, f'{rb}{atk}{ds}_{ts}_log.txt')


def parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    subparsers = parser.add_subparsers(help='commands')
    exp_args(subparsers.add_parser(
        name='experiment', help='run attack experiment'))
    vld_args(subparsers.add_parser(
        name='validate', help='dataset validator'))

    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return parser.parse_args(args)


def exp_args(parser: ArgumentParser):
    parser.set_defaults(which='exp')
    parser.add_argument(
        '-d', '--dataset',
        action="store",
        default=Experiment.DEFAULT_DS,
        help=f'path to dataset  [default: {Experiment.DEFAULT_DS}]',
    )
    parser.add_argument(
        '-c', '--cls',
        action='store',
        choices=Experiment.CLASSIFIERS,
        default=Experiment.DEFAULT_CLS,
        help=f'classifier [default: {Experiment.DEFAULT_CLS}]'
    )
    parser.add_argument(
        '-a', '--attack',
        action='store',
        choices=Experiment.ATTACKS,
        help='evasion attack [default: None]'
    )
    parser.add_argument(
        '-v', '--validator',
        action='store',
        choices=Experiment.VALIDATORS,
        help='dataset validator kind [default: None]'
    )
    parser.add_argument(
        "-o", "--out",
        action='store',
        default="output",
        help="output directory [default: output]"
    )
    parser.add_argument(
        '-f', '--folds',
        type=int,
        choices=range(1, 11),
        metavar="1-10",
        help='number of k-folds splits [default: 5]',
        default=5
    )
    parser.add_argument(
        '-i', '--iter',
        type=int,
        choices=range(1, 500),
        metavar="1-500",
        help='max attack iterations',
        default=0
    )
    parser.add_argument(
        "--robust",
        action='store_true',
        help="train a robust model (xgboost only)"
    )
    parser.add_argument(
        "--capture",
        action='store_true',
        help="save generated records (original, adversarial)",
    )
    parser.add_argument(
        "--no_log",
        action='store_true',
        help="disable automatic saving of console log",
    )
    parser.add_argument(
        "--silent",
        action='store_true',
        help="disable logging output to console"
    )


def vld_args(parser: ArgumentParser):
    parser.set_defaults(which='vld')
    parser.add_argument(
        '-d', '--dataset',
        dest='dataset',
        action="store",
        help='path to dataset to validate',
    )
    parser.add_argument(
        '-v', '--validator',
        dest='validator',
        action='store',
        choices=Experiment.VALIDATORS,
        help='dataset validator kind [default: None]'
    )
    parser.add_argument(
        "--capture",
        action='store_true',
        help="save generated records (original, adversarial)",
    )
    parser.add_argument(
        "-o", "--out",
        action='store',
        default="output",
        help="output directory [default: output]"
    )
    parser.add_argument(
        '-s', "--silent",
        action='store_true',
        help="disable logging output to console"
    )


if __name__ == '__main__':
    main()
