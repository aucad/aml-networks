# Training logs

These are logged results of training on Aposemat IoT-23 dataset.

Goal: train on 50/50 split data

- test on highly benign data
- test on highly malicious data

What is the prediction accuracy in these scenarios (compared to [previous](../2-24/readme.md))?
 
Training set: CTU-IoT-Malware-Capture-1-1 (Hide and Seek)

- Source: [https://mcfp.felk.cvut.cz/.../CTU-IoT-Malware-Capture-1-1][1]
- Preprocessed: [data/CTU-IoT-Malware-Capture-1-1.csv][2]

46.5 % benign, 53.5 % malicious

| Label                     | Flows   | 
| :------------------------ | ------: |
| Benign                    | 469275  |
| Malicious                 | 539473  |

12 attributes:

- proto
- duration
- orig_bytes
- resp_bytes
- conn_state
- missed_bytes
- history
- orig_pkts
- orig_ip_bytes
- resp_pkts
- resp_ip_bytes
- label


[1]: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-1-1
[2]: https://github.com/iotcad/module-2/blob/44967f3e6aa6288a4eb806face3bf21686f89851/data/CTU-IoT-Malware-Capture-1-1.csv