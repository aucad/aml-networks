# Classifier Comparison on IoT-23 Data

These are results of training on [Aposemat IoT-23 dataset][0], using different ratios of benign/malicious 
data during training, to compare results and impact on various learning algorithms.

- **Classifiers:** AdaBoost, ANN, naive bayes, SVM, decision tree

- **Attributes (12):** proto, duration, orig_bytes, resp_bytes, conn_state, missed_bytes, history, orig_pkts,
  orig_ip_bytes, resp_pkts, resp_ip_bytes, label

- **Validation:** 10-folds cross-validation

We trained each classifier on a dataset with different ratio of benign vs. malicious examples (mostly benign, mostly
malicious, even split), then tested the trained classifier against the following test sets.

| Test datasets                 | Benign | Malicious |   Ratio | 
|:------------------------------|:------:|:---------:|--------:|
| [CTU-Honeypot-Capture-7-1][4] |  120   |     0     | 100 / 0 |
| [CTU-Malware-Capture-44-1][2] |  211   |    26     | 90 / 10 | 
| [CTU-Malware-Capture-8-1][5]  | 2181	  |   8222    | 20 / 80 | 

The purpose of this analysis was to identify suitable classifier, and their behavior on different splits of
training/testing data.

**Results**

[Benign Training](#malware-capture-20-1-mostly-benign) |
[Malicious Training](#malware-capture-34-1-mostly-malicious) |
[Even Split](#malware-capture-1-1-even-split)

---

## Malware Capture 20-1 (mostly benign)

| Training dataset              | Benign | Malicious |      Ratio | 
|:------------------------------|:------:|:---------:|-----------:|
| [CTU-Malware-Capture-20-1][1] |  3193  |    16     | 99.5 / 0.5 |

**Results**

FP rate, precision, recall are for the malicious class

| Method               |             | Training<br/>Benign | Honeypot-7<br/>(All Benign) | Malware-44<br/>(Mostly Benign) | Malware-8<br/>(Mostly Malicious) |
|:---------------------|-------------|--------------------:|----------------------------:|-------------------------------:|---------------------------------:|
| [Adaboost][AB0]      | — Accuracy  |           99.9377 % |                   99.2308 % |                      94.5148 % |                       100.0000 % |
|                      | — FP Rate   |               0.000 |                       0.008 |                          0.000 |                            0.000 |
|                      | — Precision |               0.938 |                       0.000 |                          1.000 |                            1.000 |
|                      | — Recall    |               0.938 |                         N/A |                          0.500 |                            1.000 |
|                      |             |                     |                             |                                |                                  |
| [ANN][AN0]           | — Accuracy  |           30.1963 % |                         0 % |                      10.9705 % |                        79.0349 % |
|                      | — FP Rate   |               0.700 |                       1.000 |                          1.000 |                            1.000 |
|                      | — Precision |               0.005 |                       0.000 |                          0.110 |                            0.790 |
|                      | — Recall    |               0.688 |                         N/A |                          1.000 |                            1.000 |
|                      |             |                     |                             |                                |                                  |
| [Decision tree][DT0] | — Accuracy  |           99.9688 % |                   92.3077 % |                      98.7342 % |                        99.9808 % |
|                      | — FP Rate   |               0.000 |                       0.077 |                          0.009 |                            0.000 |
|                      | — Precision |               0.941 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |               1.000 |                         N/A |                          0.962 |                            1.000 |
|                      |             |                     |                             |                                |                                  |
| [Naive Bayes][NB0]   | — Accuracy  |           99.3768 % |                   55.3846 % |                      94.0928 % |                        20.9459 % |
|                      | — FP Rate   |               0.003 |                       0.446 |                          0.005 |                            0.001 |
|                      | — Precision |               0.357 |                       0.000 |                          0.929 |                            0.000 |
|                      | — Recall    |               0.313 |                         N/A |                          0.500 |                            0.000 |
|                      |             |                     |                             |                                |                                  |
| [SVM][SV0]           | — Accuracy  |           99.9377 % |                   99.2308 % |                      99.5781 % |                       100.0000 % |
|                      | — FP Rate   |               0.000 |                       0.008 |                          0.000 |                            0.000 |
|                      | — Precision |               0.938 |                       0.000 |                          1.000 |                            1.000 |
|                      | — Recall    |               0.938 |                         N/A |                          0.962 |                            1.000 |

**Decision tree**

![img](logs/20-1-tree.png)

<br/><br/>

## Malware Capture 34-1 (mostly malicious)

| Training dataset              | Benign | Malicious |  Ratio | 
|-------------------------------|:------:|:---------:|-------:|
| [CTU-Malware-Capture-34-1][3] |  1923  |   21222   | 8 / 92 |

**Results**

FP rate, precision, recall are for the malicious class

| Method               |             | Training<br/>Malicious | Honeypot-7<br/>(All Benign) | Malware-44<br/>(Mostly Benign) | Malware-8<br/>(Mostly Malicious) |
|:---------------------|:------------|-----------------------:|----------------------------:|-------------------------------:|---------------------------------:|
| [Adaboost][AB1]      | — Accuracy  |              99.5118 % |                   54.6154 % |                      98.7342 % |                        99.9808 % |
|                      | — FP Rate   |                  0.058 |                       0.454 |                          0.009 |                            0.001 |
|                      | — Precision |                  0.995 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |                  1.000 |                         N/A |                          0.962 |                            1.000 |
|                      |             |                        |                             |                                |                                  |
| [ANN][AN1]           | — Accuracy  |              91.8859 % |                   10.7692 % |                       6.7511 % |                        79.0349 % |
|                      | — FP Rate   |                  0.954 |                       0.892 |                          0.995 |                            1.000 |
|                      | — Precision |                  0.920 |                       0.000 |                          0.067 |                            0.790 |
|                      | — Recall    |                  0.998 |                         N/A |                          0.577 |                            1.000 |
|                      |             |                        |                             |                                |                                  |
| [Decision tree][DT1] | — Accuracy  |              99.5766 % |                   54.6154 % |                      94.5148 % |                        99.9808 % |
|                      | — FP Rate   |                  0.050 |                       0.454 |                          0.005 |                            0.001 |
|                      | — Precision |                  0.995 |                       0.000 |                          0.933 |                            1.000 |
|                      | — Recall    |                  1.000 |                         N/A |                          0.538 |                            1.000 |
|                      |             |                        |                             |                                |                                  |
| [Naive Bayes][NB1]   | — Accuracy  |              99.5463 % |                   46.9231 % |                      94.9367 % |                        99.9808 % |
|                      | — FP Rate   |                  0.052 |                       0.531 |                          0.000 |                            0.001 |
|                      | — Precision |                  0.995 |                       0.000 |                          1.000 |                            1.000 |
|                      | — Recall    |                  1.000 |                         N/A |                          0.538 |                            1.000 |
|                      |             |                        |                             |                                |                                  |
| [SVM][SV1]           | — Accuracy  |              99.5723 % |                   53.8462 % |                      94.9367 % |                        99.9808 % |
|                      | — FP Rate   |                  0.051 |                       0.462 |                          0.005 |                            0.001 |
|                      | — Precision |                  0.995 |                       0.000 |                          0.938 |                            1.000 |
|                      | — Recall    |                  1.000 |                         N/A |                          0.577 |                            1.000 |

**Decision tree**

![img](logs/34-1-tree.png)

<br/><br/>

## Malware Capture 1-1 (even split)

| Training dataset                 | Benign | Malicious |         Ratio | 
|:---------------------------------|:------:|:---------:|--------------:|
| [CTU-IoT-Malware-Capture-1-1][6] | 117318 |  134868   |   46.5 / 53.5 |

**Results**

FP rate, precision, recall are for the malicious class

| Method               |             | Training<br/>(50/50) | Honeypot-7<br/>(All Benign) | Malware-44<br/>(Mostly Benign) | Malware-8<br/>(Mostly Malicious) |
|:---------------------|-------------|---------------------:|----------------------------:|-------------------------------:|---------------------------------:|
| [Adaboost][AB2]      | — Accuracy  |            95.6528 % |                   92.3077 % |                      98.7342 % |                        99.9808 % |
|                      | — FP Rate   |                0.000 |                       0.077 |                          0.009 |                            0.001 |
|                      | — Precision |                1.000 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |                0.907 |                         N/A |                          0.926 |                            1.000 |
|                      |             |                      |                             |                                |                                  |
| [ANN][AN2]           | — Accuracy  |            95.2777 % |                   89.2308 % |                      98.7342 % |                        99.9808 % |
|                      | — FP Rate   |                0.101 |                       0.108 |                          0.009 |                            0.001 |
|                      | — Precision |                0.919 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |                1.000 |                         N/A |                          0.926 |                            1.000 |
|                      |             |                      |                             |                                |                                  |
| [Decision tree][DT2] | — Accuracy  |            95.6726 % |                   99.2308 % |                      98.7342 % |                       100.0000 % |
|                      | — FP Rate   |                0.093 |                       0.008 |                          0.009 |                            0.000 |
|                      | — Precision |                0.925 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |                1.000 |                         N/A |                          0.962 |                            1.000 |
|                      |             |                      |                             |                                |                                  |
| [Naive Bayes][NB2]   | — Accuracy  |            63.7323 % |                   25.3846 % |                      90.7173 % |                        40.7094 % |
|                      | — FP Rate   |                0.032 |                       0.746 |                          0.005 |                            0.001 |
|                      | — Precision |                0.925 |                       0.000 |                          0.833 |                            0.999 |
|                      | — Recall    |                0.350 |                         N/A |                          0.192 |                            0.250 |
|                      |             |                      |                             |                                |                                  |
| [SVM][SV2]           | — Accuracy  |            95.6782 % |                   93.0769 % |                      98.7342 % |                        99.9808 % |
|                      | — FP Rate   |                0.093 |                       0.069 |                          0.009 |                            0.001 |
|                      | — Precision |                0.925 |                       0.000 |                          0.926 |                            1.000 |
|                      | — Recall    |                1.000 |                         N/A |                          0.962 |                            1.000 |

**Decision tree**

![img](logs/1-1-tree.png)

[0]: https://github.com/iotcad/sensor-data/tree/main/iot-23
[1]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-20-1-labeled.csv
[2]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-44-1-labeled.csv
[3]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-34-1-labeled.csv
[4]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Honeypot-Capture-7-1-labeled.csv
[5]: https://github.com/iotcad/sensor-data/blob/de0d85ec49f0e3560e2715abe5d7fcb48194be24/iot-23/CTU-Malware-Capture-8-1-labeled.csv
[6]: https://github.com/iotcad/sensor-data/blob/de0d85ec49f0e3560e2715abe5d7fcb48194be24/iot-23/12-attr/CTU-IoT-Malware-Capture-1-1-sampled.csv

[AB0]: logs/20-1-adaboost
[AN0]: logs/20-1-ann
[DT0]: logs/20-1-tree
[NB0]: logs/20-1-bayes
[SV0]: logs/20-1-svm
[AB1]: logs/34-1-adaboost
[AN1]: logs/34-1-ann
[DT1]: logs/34-1-tree
[NB1]: logs/34-1-bayes
[SV1]: logs/34-1-svm
[NB2]: logs/1-1-bayes
[DT2]: logs/1-1-tree
[SV2]: logs/1-1-svm
[AN2]: logs/1-1-ann
[AB2]: logs/1-1-adaboost