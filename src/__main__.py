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
from argparse import ArgumentParser
from sys import exit
from typing import Optional, List

from src import __version__, __title__, \
    Experiment, Validator, Plot, utility


def main():
    """
    Run adversarial ML attacks and defenses on tree-based classifiers
    on network data.
    """
    parser = ArgumentParser(prog=__title__, description=main.__doc__)
    args = parse_args(parser)

    options = ['exp', 'vld', 'plot']
    choice = [args.which if hasattr(args, 'which') else None] * len(options)
    is_exp, is_vld, is_plot = [a == b for a, b in zip(choice, options)]
    init_logger(logging.FATAL if args.silent else logging.DEBUG)
    if hasattr(args, 'out'):
        utility.ensure_dir(args.out)

    if is_plot:
        Plot(args.dir, args.format)
        return

    if not args.dataset:
        parser.print_help()
        exit(1)

    if is_vld:
        Validator.validate_dataset(
            args.validator, args.dataset, args.capture, args.out)

    if is_exp:
        Experiment(utility.ts_str(), **args.__dict__).run()


def init_logger(level: int, fn: str = None):
    """Create a logger instance"""
    fmt = "[%(asctime)s]: %(message)s"
    date_fmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    logger = logging.getLogger(__title__)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if fn is not None:
        file_handler = logging.FileHandler(filename=f'{fn}')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    subparsers = parser.add_subparsers()
    exp_args(subparsers.add_parser(
        name='experiment', help='run attack experiment'))
    validator_args(subparsers.add_parser(
        name='validate', help='dataset validator'))
    plot_args(subparsers.add_parser(
        name='plot', help='plot experiment results'))

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
        choices=range(2, 11),
        metavar="2-10",
        help='number of k-folds splits [default: 5]',
        default=5
    )
    parser.add_argument(
        '-i', '--iter',
        type=int,
        choices=range(0, 500),
        metavar="1-500",
        help='max attack iterations',
        default=0
    )
    parser.add_argument(
        '-s', '--sample_size',
        type=int,
        help='number of records to perturb (all if unset)',
        default=0
    )
    parser.add_argument(
        '-t', '--sample_times',
        type=int,
        choices=range(1, 10),
        metavar="1-10",
        help='number of times to sample [default: 1]',
        default=1
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
        "--no_save",
        action='store_true',
        help="disable save result to file",
    )
    parser.add_argument(
        "--silent",
        action='store_true',
        help="disable console log"
    )


def validator_args(parser: ArgumentParser):
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
        help="save invalid records to file",
    )
    parser.add_argument(
        "-o", "--out",
        action='store',
        default="output",
        help="output directory [default: output]"
    )
    parser.add_argument(
        "--silent",
        action='store_true',
        help="disable console log"
    )


def plot_args(parser: ArgumentParser):
    parser.set_defaults(which='plot')
    parser.add_argument(
        dest="dir",
        action='store',
        default="output",
        help="path to directory with results [default: output]"
    )
    parser.add_argument(
        '-f', '--format',
        action='store',
        choices=['plain', 'tex'],
        help='evasion attack [default: plain]'
    )
    parser.add_argument(
        "--silent",
        action='store_true',
        help="disable console log"
    )


if __name__ == '__main__':
    main()
