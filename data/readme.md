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
| nb15    | UNSW-NB15 > 175K training set            |   ✔️    | 10000 |   50 % |      50 % |

To evaluate additional data sources, make sure the input data is fully numeric, and in csv format.
Use these existing data sets as examples.

### Preprocessing steps

Same steps apply to both IoT-23 and UNSW-NB15 data. 
These steps are easy to do with [Weka](https://waikato.github.io/weka-wiki/downloading_weka/).

1. Remove irrelevant features not included in the feature table below.

2. For each remaining nominal attribute:
   - Check if they have many values, e.g., `history` for IoT-23.
   - Merge low-frequency values (`MergeManyValues` filter in Weka)
   - After this, at most 2 or 3 distinct values should remain for nominal attributes.

3. One hot encode nominal attributes (`NomialToBinary` filter in Weka)

4. Change class labels to numeric values: Benign = `0` and Malicious = `1`.

5. Missing values should be actual nulls (not `-` `?` etc.).

6. Use Weka's supervised instance `SpreadSubsample` with `distributionSpread=1.0`, 
   to obtain a random sample of desired size, with equal class distribution 
   (see [details here](https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/)).

7. Save as a comma-separated file. The expected input is CSV-format.

### Dataset Features

Data set features used in experimental evaluation.

| #   | Feature            | Description                       |
|-----|--------------------|-----------------------------------|
|     | **IOT-23**         |                                   | 
| 1.  | `proto `           | Network transaction protocol.     |
| 2.  | `duration`         | Duration of transmission.         |
| 3.  | `orig_bytes `      | Data sent to the device.          |
| 4.  | `resp_bytes `      | Data sent by the device.          |
| 5.  | `conn_state `      | State of the connection.          |
| 6.  | `missed_bytes `    | Missed bytes in a message.        |
| 7.  | `history `         | History of connection state.      |
| 8.  | `orig_pkts`        | Packets sent to the device.       |
| 9.  | `orig_ip_bytes `   | Bytes sent to the device.         |
| 10. | `resp_pkts`        | Packets sent from the device.     |
| 11. | `resp_ip_bytes `   | Bytes sent from the device.       |
| 12. | `label `           | Type of capture.                  |
|     |                    |                                   |
|     | **UNSW-NB15**      |                                   | 
 | 1.  | `proto`            | Network transaction protocol.     |
 | 2.  | `state`            | State and dependent protocol.     |
 | 3.  | `service`          | Http, ftp, smtp, ssh, ... or (-). |
 | 4.  | `dur`              | Record total duration.            |
 | 5.  | `sbytes`           | Src to dest transaction bytes.    |
 | 6.  | `dbytes`           | Dest to src transaction bytes.    |
 | 7.  | `sloss`            | Retransmitted/dropped by src.     |
 | 8.  | `dloss`            | Retransmitted/dropped by dest.    |
 | 9.  | `swin`             | Src TCP window adv. value.        |
 | 10. | `dwin`             | Dest TCP window adv. value.       |
 | 11. | `stcpb`            | Src TCP base seq. number.         |
 | 12. | `dtcpb`            | Dest TCP base seq. number.        |
 | 13. | `smeansz`          | Src mean packet size.             |
 | 14. | `dmeansz`          | Dest mean packet size.            |
 | 15. | `sjit`             | Src jitter (mSec).                |
 | 16. | `djit`             | Dest jitter (mSec).               |
 | 17. | `trans_depth`      | Pipelined depth of transaction.   |
 | 18. | `ct_srv_src`       | Same service and src address.     |
 | 19. | `ct_srv_dst`       | Same service and dest address.    |
 | 20. | `ct_src_ ltm`      | Same src address.                 |
 | 21. | `ct_dst_ltm`       | Same dest address.                |
 | 22. | `ct_src_dport_ltm` | Same src addr. and dest port.     |
 | 23. | `ct_dst_sport_ltm` | Same dest addr. and src port.     |
 | 24. | `ct_dst_src_ltm`   | Same src and dest address.        |
 | 25. | `label`            | Type of capture.                  |

