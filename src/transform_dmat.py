from os import path
from random import sample
from sys import argv, exit

from utility import read_csv, save_file, color_text as c, int_label

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


def transform_row(row):
    """Format 1 row of CSV data as DMatrix data"""
    label, data, result = row[-1], row[0:-2], []
    if not str(label).isdigit():
        label = int_label(label)
    result.append(label)
    for index, value in enumerate(data):
        if not str(value).isnumeric():
            continue
        result.append(f'{index}:{value}')

    return " ".join([str(s) for s in result])


def out_filename(file_in, split_name):
    """Generate output filename from the input file name."""
    dir_path = path.dirname(file_in)
    filename = path.basename(path.splitext(file_in)[0])
    return path.join(dir_path, f'{filename}.{split_name}.dmat')


def split_sample(rows, test_set):
    """Split to training and testing data, where test_set is the
       percentage split in range 0.0 - 1.0"""
    n_rows = len(rows)
    sample_size = int(n_rows * test_set)
    all_indices = list(range(n_rows))
    idx = sample(all_indices, sample_size)
    test, train = [], []
    for i, row in enumerate(rows):
        target = test if i in idx else train
        target.append(row)
    return train, test


def save_sampled_lines(file_in, split_name, lines):
    out_file = out_filename(file_in, split_name)
    save_file(out_file, lines)
    return out_file


def convert(file_in):
    """Convert CSV file to DMatrix file."""
    rows = read_csv(file_in)[1]
    lines = [transform_row(r) for r in rows]
    train, test = split_sample(lines, 0.1)
    train_file = save_sampled_lines(file_in, 'train', train)
    test_file = save_sampled_lines(file_in, 'test', test)
    print(f'Converted CSV -> DMatrix ({len(rows)}/{len(lines)} rows)')
    print(f'Saved result to {c(train_file)} and {c(test_file)}')


if __name__ == '__main__':
    if len(argv) >= 2:
        convert(argv[1])
    else:
        print('Input file path is a required argument')
        exit(1)
