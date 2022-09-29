import logging
from os import path, makedirs

import numpy as np
from colorama import Fore, Style  # terminal colors

logger = logging.getLogger(__name__)

DEFAULT_DS = 'data/CTU-1-1.csv'
"""When no dataset is defined, use this one by default."""
"""We use here a dataset with ~ 50/50 split"""

RESULT_DIR = 'output'
"""Directory for writing outputs"""


def ensure_out_dir(dir_path):
    return path.exists(dir_path) or makedirs(dir_path)


def text_label(i):
    """convert text label to numeric"""
    return 'malicious' if i == 1 else 'benign'


def int_label(text):
    """convert numeric label to text label"""
    return 0 if text.lower() == 'benign' else 1


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


def int_cols(num_mat):
    """Finds integer valued attributes in a 2d matrix"""
    indices = []
    for col_i in range(len(num_mat[0])):
        if set(list(np.unique(num_mat[:, col_i]))).issubset({0, 1}):
            indices.append(col_i)
    return indices


def freeze_types(num_mat):
    return int_cols(num_mat)


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


def dump_result(evasions, train_x, train_y, adv_x, adv_y, attr):
    """Write to csv file original and adversarial examples.

    arguments:
        evasions - list of indices where attack succeeded
        train_x - original training data, np.array (2d)
        train_y - original labels, np.array (1d)
        adv_x - adversarial examples, np.array (2d)
        adv_y - adversarial labels, np.array (1d)
        attr - data attributes
    """

    import csv

    def fmt(x, y):
        # append row and label, for each row
        labels = y[evasions].reshape(-1, 1)
        return (np.append(x[evasions, :], labels, 1)).tolist()

    ensure_out_dir(RESULT_DIR)
    inputs = [[fmt(train_x, train_y), 'ori.csv'],
              [fmt(adv_x, adv_y), 'adv.csv']]

    # include label column
    int_values = int_cols(train_x)

    for (rows, name) in inputs:
        with open(path.join(RESULT_DIR, name),
                  'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            w.writerow(attr)
            for row in rows:
                fmt_row = []
                for i, val in enumerate(row):
                    if i in int_values or i == len(row) - 1:
                        fmt_row.append(int(val))
                    else:
                        fmt_row.append(val)
                w.writerow(fmt_row)
