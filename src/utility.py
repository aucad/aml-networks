from csv import reader
from typing import Tuple


def read_csv(path: str) -> Tuple[list, list]:
    """read some CSV file and split into headers and rows

    Arguments:
        path: file path to a CSV file

    Returns:
        Tuple where first item is data headers (list)
        and second is row of data
    """
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        all_rows = [row for row in csv_reader]
        header = all_rows[0]
        rows = all_rows[1:]

    return header, rows


def save_csv(content: str, filename: str):
    pass
