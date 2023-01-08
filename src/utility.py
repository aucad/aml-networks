import csv
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def name_only(file):
    return Path(file).stem


def generate_name(ts, args, extension=None):
    ds = name_only(args.dataset)
    rb = f'robust_{"T" if args.robust else "F"}_' \
        if hasattr(args, 'robust') else ''
    i = f'_i{args.iter}' if args.iter > 0 else ''
    atk = f'{args.attack}_' if hasattr(args, 'attack') else ''
    ext = f'.{extension}' if extension else ''
    return os.path.join(args.out, f'{rb}{atk}{ds}{i}_{ts}{ext}')


def show(label: str, value: Any):
    """Pretty print labelled text with alignment.

    Arguments:
        label - description of the text
        value - the text to print
    """
    str_v = str(value)

    # wrap values lines
    wrap_size, label_w, log_pad = 512, 18, 12
    chunks, chunk_size, lines = len(str_v), wrap_size, []
    if chunks < chunk_size and str_v.find("\n") < 0:
        lines = [str_v]
    else:
        rest = str_v
        while len(rest) > 0:
            newline = rest[:chunk_size].find("\n")
            if newline < 0 and len(rest) < chunk_size:
                i = len(rest)
            else:
                space = rest[:chunk_size].rfind(" ")
                i = newline if newline > 0 else \
                    (space if space > (chunk_size // 2)
                     else chunk_size)
            line, remaining = rest[:i].strip(), rest[i:].strip()
            lines.append(line)
            rest = remaining
    fmt_lines = "\n".join(
        [(' ' * (label_w + log_pad) if i > 0 else '')
         + s for i, s in enumerate(lines)])

    text = f'{label} '.ljust(label_w, '-') + fmt_lines
    logger.debug(text)


def show_ratio(label, num, denom):
    if denom == 0:
        return show(label, 0)
    ratio = 100 * num / denom
    return show(label, f'{round(num, 0)} of {round(denom, 0)}'
                       f' - {ratio:.1f} %')


def clear_one_line():
    """Clear previous line of terminal output."""
    cols = 256
    print("\033[A{}\033[A".format(' ' * cols), end='\r')


def ensure_dir(dir_path: str):
    """Make sure a directory exists.

    Arguments:
        dir_path - path to a directory
    """
    return os.path.exists(dir_path) or os.makedirs(dir_path)


def attr_fix(attrs):
    """Remove selected special chars from attributes so that
    the remaining forms a valid Python identifier."""
    return [a.replace(' ', '')
            .replace('=', '_')
            .replace('-', '')
            .replace('^', '_')
            .replace('conn_state_other', 'conn_state_OTH')
            for a in attrs]


def ts_str(length: int = 4) -> str:
    """Make a string of current timestamp.

    Arguments:
        length - number of digits to keep (from the smallest unit)
    """
    return str(round(time.time() * 1000))[-length:]


def sdiv(num, denom):
    """Save division."""
    return 0 if denom == 0 else num / denom


def dump_num_dict(reasons):
    reason_pairs = [(v, f'{v} * {k}')
                    for k, v in reasons.items() if v > 0]
    sorted_reasons = sorted(reason_pairs, reverse=True)
    return '\n'.join([txt for _, txt in sorted_reasons])


def read_dataset(dataset_path):
    df = pd.read_csv(dataset_path).fillna(0)
    attrs = attr_fix([col for col in df.columns])
    rows = np.array(df)
    return attrs, rows


def write_dataset(file_name, attrs, rows, int_cols=None):
    with open(file_name, 'w', newline='') as fp:
        w = csv.writer(fp, delimiter=',')
        w.writerow(attrs)
        w.writerows([
            [int(val) if i in (int_cols or []) else val
             for i, val in enumerate(row)]
            for row in rows])
