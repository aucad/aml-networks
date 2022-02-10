import sys
import collections
from random import shuffle
from datetime import datetime
from os import path

from utility import read_csv, save_csv

"""
This is a utility script for concatenating two csv files into one. 

Arguments:
    input_1: path to CSV file
    input_2: path to second CSV file
"""


def normalize_labels(rows, idx):
    """make labels case insensitive"""
    for row in rows:
        row[idx] = (row[idx]).lower()


LABEL = 'label'
h1, r1 = read_csv(sys.argv[1])
h2, r2 = read_csv(sys.argv[2])
DT_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = f'dataset_{DT_STR}.csv'
OUT_DIR = 'data'

if h1 != h2:
    print('(!) Headers do not match, cannot merge incompatible data')
    print('headers 1: ', ','.join(h1))
    print('headers 2: ', ','.join(h2))
    sys.exit(1)

# merge rows and find index of label column
ALL_ROWS, LABEL_IDX = r1 + r2, h1.index(LABEL)
TROWS = len(ALL_ROWS)

# make all labels same letter case
normalize_labels(ALL_ROWS, LABEL_IDX)

CLASS_LABELS = [r[LABEL_IDX] for r in ALL_ROWS]
LABEL_FREQ = collections.Counter(CLASS_LABELS)

# print some stats on each label
for label in list(set(CLASS_LABELS)):
    freq = LABEL_FREQ[label]
    perc = 100 * freq / TROWS
    print(f'{label.ljust(10)} ',
          f'{str(freq).rjust(10)}',
          f'{perc:.2f}'.rjust(10), '%')
print(f'Total rows: {TROWS}')

shuffle(ALL_ROWS)  # maybe these need to be in chronological order?

save_csv(path.join(OUT_DIR, FILENAME), ALL_ROWS, h1)
print('done.')