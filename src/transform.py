from sys import argv

from utility import read_csv, save_csv

FILE_IN, FILE_OUT = argv[1], argv[2]
SEP, COMMENT = '   ', '#'

head, rows = read_csv(FILE_IN, delimiter='\t')
attrs = "proto,duration,orig_bytes,resp_bytes,conn_state," \
        "missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts," \
        "resp_ip_bytes,label".split(",")


def split_spaces(row: list):
    """some values are space delimited, apply custom split."""
    return [j for sub in
            [item.split(SEP) if SEP in item else [item]
             for item in row] for j in sub]


def replace_missing(row: list):
    """Replace - with None value."""
    return [None if c == '-' else c for c in row]


def keep(row, index_list):
    """remove attributes not in attrs."""
    return [row[x] for x in index_list]


all_headers = split_spaces(head)
indices = [all_headers.index(attr) for attr in attrs]

csv_headers = keep(all_headers, indices)
csv_rows = [keep(replace_missing(split_spaces(row)), indices)
            for row in rows[:-1]]

save_csv(FILE_OUT, csv_rows, csv_headers)

print(f'read {len(rows)} rows')
print(f'wrote {len(csv_headers)} headers and {len(csv_rows)} rows')
