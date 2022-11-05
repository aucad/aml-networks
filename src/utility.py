import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def show(label: str, value: Any):
    """Pretty print labelled text with alignment.

    Arguments:
        label - description of the text
        value - the text to print
    """
    str_v = str(value)

    # wrap values lines
    wrap_size, label_w, log_pad = 512, 18, 0
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


def ts_str(length: int = 4) -> str:
    """Make a string of current timestamp.

    Arguments:
        length - number of digits to keep (from the smallest unit)
    """
    return str(round(time.time() * 1000))[-length:]
