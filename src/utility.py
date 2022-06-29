from csv import reader, writer
from typing import Tuple, List

import numpy as np
import pandas as pd
from colorama import Fore, Style  # terminal colors
from sklearn.model_selection import train_test_split

DEFAULT_DS = 'data/CTU-44-1.csv'


def text_label(i):
    return 'benign' if i == 1 else 'malicious'


def int_label(text):
    return 1 if text.lower() == 'benign' else 0


def show(msg, value, end='\n'):
    print(f'{msg} '.ljust(30, '-'), color_text(value), end=end)


def normalize(data):
    """normalize values in range 0.0 - 1.0."""
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(len(data[0])):
        data[:, i] = (data[:, i]) / max(data[:, i])
        data[:, i] = np.nan_to_num(data[:, i])
    return data


def load_csv_data(dataset_path, test_size=0.1):
    """Read dataset and split to train/test using random sampling."""

    df = pd.read_csv(dataset_path)
    attrs = [col for col in df.columns]
    split = 0 < test_size < len(df)
    test_x, test_y = np.array([]), np.array([])

    # sample training/test instances
    if split:
        train, test = train_test_split(df, test_size=test_size)
        test_x = normalize(np.array(test)[:, :-1])
        test_y = np.array(test)[:, -1].astype(int).flatten()
    else:
        train = df

    train_x = normalize(np.array(train)[:, :-1])
    train_y = np.array(train)[:, -1].astype(int).flatten()
    classes = np.unique(train_y)

    return attrs, classes, train_x, train_y, test_x, test_y


def score(true_labels, predictions, positive=0, display=False):
    """Calculate performance metrics."""
    sc, tp_tn, num_pos_pred, num_pos_actual = 0, 0, 0, 0
    for actual, pred in zip(true_labels, predictions):
        int_pred = int(round(pred, 0))
        if int_pred == positive:
            num_pos_pred += 1
        if actual == positive:
            num_pos_actual += 1
        if int_pred == actual:
            tp_tn += 1
        if int_pred == actual and int_pred == positive:
            sc += 1

    accuracy = tp_tn / len(predictions)
    precision = 1 if num_pos_pred == 0 else sc / num_pos_pred
    recall = 1 if num_pos_actual == 0 else sc / num_pos_actual
    f_score = (2 * precision * recall) / (precision + recall)

    if display:
        show('Accuracy', f'{accuracy * 100:.2f} %')
        show('Precision', f'{precision * 100:.2f} %')
        show('Recall', f'{recall * 100:.2f} %')
        show('F-score', f'{f_score * 100:.2f} %')

    return accuracy, precision, recall, f_score


def color_text(text):
    """Display terminal text in color."""
    return Fore.GREEN + str(text) + Style.RESET_ALL


def binary_attributes(np_array):
    """Get column indices of binary attributes"""
    return [feat for feat in range(len(np_array[0]))
            if len(list(set(np_array[:, feat]))) == 2]


def non_bin_attributes(np_array):
    """Get column indices of non-binary attributes"""
    return [feat for feat in range(len(np_array[0]))
            if len(list(set(np_array[:, feat]))) > 2]


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


def read_lines(path: str) -> List[str]:
    """Read text file line by line"""
    with open(path, 'r') as fp:
        return fp.readlines()


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
