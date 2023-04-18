# Datasets

We consider two network traffic captures:

1. [Aposemat IoT-23](https://www.stratosphereips.org/datasets-iot23/) is a labeled dataset with malicious and benign IoT
   network traffic.

2. [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) is a network intrusion dataset that contains
   nine different attacks.

This directory includes pre-processed, small datasets, ready for experiments.

| Dataset | Origin                                   | Sampled |  Rows | Benign | Malicious | 
|:--------|:-----------------------------------------|:-------:|------:|-------:|----------:|
| CTU     | IoT-23 > Malware Captures 1-1, 8-1, 34-1 |   ✔️    | 10000 |   50 % |      50 % |
| CTU 1-1 | IoT-23 > CTU-Malware-Capture-1-1         |   ✔️    | 10000 |   50 % |      50 % |
| nb15    | UNSW-NB15 > 175K training set            |   ✔️    | 10000 |   50 % |      50 % |

To evaluate additional data sources, make sure the input data is fully numeric, and in csv format.
Use these existing data sets as examples.
