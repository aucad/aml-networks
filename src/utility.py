import numpy as np
import pandas as pd
from colorama import Fore, Style  # terminal colors
from sklearn.model_selection import train_test_split

DEFAULT_DS = 'data/CTU-1-1.csv'
"""When no dataset is defined, use this one by default."""
"""We use here a dataset with ~ 50/50 split"""


def text_label(i):
    """convert text label to numeric"""
    return 'malicious' if i == 1 else 'benign'


def int_label(text):
    """convert numeric label to text label"""
    return 0 if text.lower() == 'benign' else 1


def show(msg, value, end='\n'):
    """Pretty print output with colors and alignment"""
    print(f'{msg} '.ljust(30, '-'), color_text(value), end=end)


def normalize(data):
    """normalize values in range 0.0 - 1.0."""
    np.seterr(divide='ignore', invalid='ignore')
    for i in range(len(data[0])):
        data[:, i] = (data[:, i]) / max(data[:, i])
        data[:, i] = np.nan_to_num(data[:, i])
    return data


def load_csv_data(dataset_path, test_size=0.1, max=-1):
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
    if max > 0:
        train_x, train_y = train_x[: max, :], train_y[: max]

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

    inputs = [[fmt(train_x, train_y), 'ori.csv'],
              [fmt(adv_x, adv_y), 'adv.csv']]

    for (rows, name) in inputs:
        with open(name, 'w', newline='') as csvfile:
            w = csv.writer(csvfile, delimiter=',')
            w.writerow(attr)
            w.writerows(rows)
