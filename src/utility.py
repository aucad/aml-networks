from csv import reader, writer
from typing import Tuple, List

from colorama import Fore, Style  # terminal colors


def color_text(text):
    """Display terminal text in color."""
    return Fore.GREEN + str(text) + Style.RESET_ALL


def read_csv(path: str, delimiter=',') -> Tuple[list, list]:
    """read some CSV file and split into headers and rows

    Arguments:
        path: file path to a CSV file
        delimiter: optional delimiter

    Returns:
        Tuple where first item is data headers (list)
        and second is row of data
    """
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj, delimiter=delimiter)
        all_rows = [row for row in csv_reader]
        header = all_rows[0]
        rows = all_rows[1:]

    return header, rows


def save_csv(filename: str, rows: List[List], headers: List = None):
    """saves CSV file

    Arguments:
        filename: where to save file (with path and extension)
        rows: list of data rows
        headers: csv file headers (optional)
    """
    with open(filename, 'w') as fp:
        w = writer(fp)

        if headers:
            w.writerow(headers)

        for row in rows:
            w.writerow(row)


def save_file(filename: str, lines: List[str]):
    """Write a text file.

    Arguments:
        filename: where to save file (with path and extension)
        lines: lines fo text
    """
    with open(filename, 'w') as fp:
        fp.write('\n'.join(lines))
