import sys
import collections
from random import shuffle, randint
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


f1 = sys.argv[1]
f2 = sys.argv[2]
LABEL = 'label'
h1, r1 = read_csv(f1)
h2, r2 = read_csv(f2)
DT_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = f'dataset_{DT_STR}.csv'
OUT_DIR = 'data'

if h1 != h2:
    print('(!) Headers do not match, cannot merge incompatible data')
    print('headers 1: ', ','.join(h1))
    print('headers 2: ', ','.join(h2))
    sys.exit(1)

# merge rows and find index of label column
ALL_ROWS, LABEL_IDX = r1, h1.index(LABEL)
TROWS = len(ALL_ROWS)

# make all labels same letter case
normalize_labels(ALL_ROWS, LABEL_IDX)
ALL_ROWS.sort(key=lambda the_list: the_list[LABEL_IDX])

CLASS_LABELS = [r[LABEL_IDX] for r in ALL_ROWS]
LABEL_FREQ = collections.Counter(CLASS_LABELS)
MIN_FREQ = min([LABEL_FREQ[label] for label in list(set(CLASS_LABELS))])
SAMPLED_LIST = []

# print some stats on each label
print(f'Total rows: {TROWS}')
print(f'Min freq: {MIN_FREQ}')

for label in list(set(CLASS_LABELS)):
    freq = LABEL_FREQ[label]
    perc = 100 * freq / TROWS
    subset = [record for record in ALL_ROWS if
              record[LABEL_IDX] == label]
    RAND_FACTOR = randint(-int(MIN_FREQ * .05), int(MIN_FREQ * .05))
    shuffle(subset)
    # take uniformly with some randomness
    take = subset[:(MIN_FREQ + RAND_FACTOR)]
    SAMPLED_LIST += take

    print(f'{label.ljust(10)} ',
          f'{str(freq).rjust(10)} ',
          f'{perc:.2f} %',
          f'took: {str(len(take))}'.rjust(20))

TROWS = len(SAMPLED_LIST)
CLASS_LABELS = [r[LABEL_IDX] for r in SAMPLED_LIST]
LABEL_FREQ = collections.Counter(CLASS_LABELS)
shuffle(SAMPLED_LIST)

for label in list(set(CLASS_LABELS)):
    freq = LABEL_FREQ[label]
    perc = 100 * freq / TROWS
    print(f'{label.ljust(10)} ',
          f'{str(freq).rjust(10)} ',
          f'{perc:.2f} %')

print(f'Total rows after sampling: {TROWS}')
save_csv(path.join(OUT_DIR, FILENAME), SAMPLED_LIST, h1)
print('done.')
