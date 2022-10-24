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
    ClsLoader, AttackLoader, Validator, DatasetLoader

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

    x, y, attrs, folds = DatasetLoader.load_csv(
        args.dataset, args.kfolds)
    cls = ClsLoader.init(
        args.cls, args.out, attrs, x, y, args.dataset)
    attack = AttackLoader \
        .load(args.attack, args.iterated, args.plot,
              args.validator, args.dataset) \
        if args.attack else None

    for i, fold in enumerate(folds):
        print(" ")
        cls.reset() \
            .load(x.copy(), y.copy(), *fold, i + 1) \
            .train(robust=args.robust)

        if args.plot:
            cls.plot()

        if args.attack:
            attack.reset() \
                .set_cls(cls) \
                .run(max_iter=args.iter)


def __parse_args(parser: ArgumentParser, args: Optional[List] = None):
    """Setup available program arguments."""

    parser.add_argument(
        '-d', '--dataset',
        action="store",
        default=DEFAULT_DS,
        help=f'path to dataset [default: {DEFAULT_DS}]',
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
        "--iterated",
        action='store_true',
        help="Run attack with increasing max_iter"
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
