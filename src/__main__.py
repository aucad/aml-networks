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
import logging as lg
from argparse import ArgumentParser
from sys import exit
from typing import Optional, List

from src import __version__, __title__, \
    ClsLoader, AttackLoader, Validator

DEFAULT_DS = 'data/CTU-1-1.csv'


def main():
    """
    Run adversarial ML attacks and defenses on tree-based classifiers
    on network data.
    """
    parser = ArgumentParser(prog=__title__, description=main.__doc__)
    args = __parse_args(parser)
    __init_logger(lg.FATAL - (0 if args.silent else 40))

    if not args.dataset:
        parser.print_help()
        exit(1)

    cls = ClsLoader \
        .load(args.out, args.cls) \
        .load(args.dataset, args.test / 100) \
        .train(robust=args.robust)

    if args.plot:
        cls.plot()

    if args.attack:
        AttackLoader.load(args.attack) \
            .set_cls(cls) \
            .set_validator(args.validator) \
            .run()


def __parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    parser.add_argument(
        '-d', '--dataset',
        action="store",
        default=DEFAULT_DS,
        help=f'path to dataset [default: {DEFAULT_DS}]',
    )
    parser.add_argument(
        '-t', '--test',
        type=int,
        choices=range(0, 100),
        metavar="0-99",
        help='test set split percentage [default: 0]',
        default=0
    )
    parser.add_argument(
        '-c', '--cls',
        action='store',
        choices=[ClsLoader.DECISION_TREE, ClsLoader.XGBOOST],
        default=ClsLoader.XGBOOST,
        help=f'Classifier to train [default: {ClsLoader.XGBOOST}]'
    )
    parser.add_argument(
        "--robust",
        action='store_true',
        help="train a robust model, xgboost only"
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        help="generate plots"
    )
    parser.add_argument(
        '-a', '--attack',
        action='store',
        choices=[AttackLoader.HOP_SKIP, AttackLoader.ZOO],
        help=f'evasion attack [default: None]'
    )
    parser.add_argument(
        '--validator',
        action='store',
        choices=[Validator.NB15, Validator.IOT23],
        help=f'dataset validator kind [default: None]'
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
        help="disable debug logging"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return parser.parse_args(args)


def __init_logger(level: int = lg.ERROR, fn: Optional[str] = None):
    """Create a logger instance"""

    fmt = lg.Formatter("[%(asctime)s]: %(message)s", datefmt="%H:%M:%S")

    logger = lg.getLogger(__title__)
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
