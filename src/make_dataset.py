import sys
import collections
from random import shuffle
from datetime import datetime
from os import path

from utility import read_csv, save_csv

"""Small utility file for concatenating iot-23 dataset files. 

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

if LABEL not in h1:
    print(f'(!) Class label "{LABEL}" not found in dataset')
    sys.exit(1)

LABEL_IDX = h1.index(LABEL)
ALL_ROWS = r1 + r2
normalize_labels(ALL_ROWS, LABEL_IDX)
TROWS = len(ALL_ROWS)
CLASS_LABELS = [r[LABEL_IDX] for r in ALL_ROWS]
UNIQUE_LABELS = list(set(CLASS_LABELS))
LABEL_FREQ = collections.Counter(CLASS_LABELS)

for label in UNIQUE_LABELS:
    freq = LABEL_FREQ[label]
    perc = 100 * freq / TROWS
    print(f'{label.ljust(10)} ',
          f'{str(freq).rjust(10)}',
          f'{perc:.2f}'.rjust(10), '%')
print(f'Total rows: {TROWS}')

shuffle(ALL_ROWS)

save_csv(path.join(OUT_DIR, FILENAME), ALL_ROWS, h1)
print('')
