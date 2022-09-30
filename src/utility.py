import logging
from os import path, makedirs

from colorama import Fore, Style  # terminal colors

logger = logging.getLogger(__name__)


def ensure_out_dir(dir_path):
    return path.exists(dir_path) or makedirs(dir_path)


def color_text(text):
    """Display terminal text in color."""
    return Fore.GREEN + str(text) + Style.RESET_ALL


def show(msg, value, end='\n'):
    """Pretty print output with colors and alignment"""
    str_v = str(value)

    # wrap values lines
    wrap_size, label_w = 70, 30
    chunks, chunk_size, lines = len(str_v), wrap_size, []
    if chunks < chunk_size:
        lines = [str_v]
    else:
        rest = str_v
        while len(rest) > 0:
            spc = rest[:chunk_size].rfind(" ")
            i = chunk_size if (spc < chunk_size // 2) else spc
            line, remaining = rest[:i].strip(), rest[i:].strip()
            lines.append(line)
            rest = remaining
    fmt_lines = "\n".join(
        [(' ' * (label_w + 1) if i > 0 else '') + color_text(s)
         for i, s in enumerate(lines)])

    print(f'{msg} '.ljust(label_w, '-'), fmt_lines, end=end)
