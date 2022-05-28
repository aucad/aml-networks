"""
Helper methods for loading and preprocessing data for classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utility import color_text as c

DEFAULT_DS = 'data/CTU-44-1.csv'


def text_label(i):
    return 'benign' if i == 1 else 'malicious'


def int_label(text):
    return 1 if text.lower() == 'benign' else 0


def show(msg, value, end='\n'):
    print(f'{msg} '.ljust(30, '-'), c(value), end=end)


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


def score(test_labels, predictions, positive=0, display=False):
    """Calculate performance metrics."""
    sc, tp_tn, num_pos_pred, num_pos_actual = 0, 0, 0, 0
    for actual, pred in zip(test_labels, predictions):
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
