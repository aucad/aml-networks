from sys import argv

from utility import read_csv, save_csv

"""
Small utility script for formatting raw IoT-23 dataset files. 

This script takes as input (in order):

- a path to a IoT-23 file
- output filename, where to write the CSV result. 

Example:

```
python transform conn.log.labelled my_data.csv 
```

NOTE: Light manual editing is required prior to calling this script.
By default IoT-23 conn.log.labelled file has following structure:

```
#separator \x09
#set_separator	,
#empty_field	(empty)
#unset_field	-
#path	conn
#open	2019-01-01-01-01-01
#fields	uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto ...
#types	time	string	addr	port
```

These headers need to be manually removed such that only header labels

```
uid	id.orig_h	id.orig_p	id.resp_h	id.resp_p	proto ...
```

remain (on line 1), followed by the data rows. After this modification
this transformation script can be applied, to convert to CSV.
"""


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
