from os import path
from sys import argv, exit

from utility import read_csv, save_file, color_text as c
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


def out_filename(file_in):
    """Generate output filename from the input file name."""
    dir_path = path.dirname(file_in)
    filename = path.basename(path.splitext(file_in)[0])
    return path.join(dir_path, f'{filename}.dmat')


def convert(file_in):
    """Convert CSV file to DMatrix file."""
    file_out = out_filename(file_in)
    rows = read_csv(file_in)[1]
    lines = [transform_row(r) for r in rows]
    save_file(file_out, lines)
    print(f'Converted CSV -> DMatrix ({len(rows)}/{len(lines)} rows)')
    print(f'Saved result to {c(file_out)}')


if __name__ == '__main__':
    if len(argv) >= 2:
        convert(argv[1])
    else:
        print('Input file path is a required argument')
        exit(1)
