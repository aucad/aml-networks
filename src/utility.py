import logging
from os import path, makedirs

from colorama import Fore, Style  # terminal colors

logger = logging.getLogger(__name__)


class BaseUtil:

    @staticmethod
    def ensure_out_dir(dir_path):
        return path.exists(dir_path) or makedirs(dir_path)

    @staticmethod
    def color_text(text):
        """Display terminal text in color."""
        return Fore.GREEN + str(text) + Style.RESET_ALL

    @staticmethod
    def show(msg, value):
        """Pretty print output with colors and alignment"""
        str_v = str(value)

        # wrap values lines
        wrap_size, label_w, log_pad = 42, 18, 12
        chunks, chunk_size, lines = len(str_v), wrap_size, []
        if chunks < chunk_size:
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
             + BaseUtil.color_text(s) for i, s in enumerate(lines)])

        text = f'{msg} '.ljust(label_w, '-') + fmt_lines
        logger.debug(text)

    @staticmethod
    def clear_one_line():
        cols = 128
        print("\033[A{}\033[A".format(' ' * cols), end='\r')
