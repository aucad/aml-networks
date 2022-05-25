from os import path
from sys import argv

from utility import read_csv, save_file
from tree import int_label

"""
Converts a CSV file to DMatrix format for XGBoost, see: 
https://xgboost.readthedocs.io/en/stable/tutorials/input_format.html

- Input: numeric CSV file; class label is assumed to be the last column
- Output: same file formatted as DMatrix data

Provide path to input file as first argument.

Usage:

```
python src/transform_dmat.py ./data/CTU-44-1.csv 
```
"""

FILE_IN = argv[1]
FPATH = path.dirname(FILE_IN)
FNAME = path.basename(path.splitext(FILE_IN)[0])
FILE_OUT = path.join(FPATH, f'{FNAME}.dmat')
_, rows = read_csv(FILE_IN)


def transform_rows(row):
    label, data, result = row[-1], row[0:-2], []
    if not str(label).isdigit():
        label = int_label(label)
    result.append(label)
    for index, value in enumerate(data):
        if not str(value).isnumeric():
            continue
        result.append(f'{index}:{value}')

    return " ".join([str(s) for s in result])


lines = [transform_rows(r) for r in rows]
save_file(FILE_OUT, lines)

print(f'{len(rows)} csv rows transformed to {len(lines)} DMatrix rows')
print(f'saved result to {FILE_OUT}')

for row in rows:
    transform_rows(row)
