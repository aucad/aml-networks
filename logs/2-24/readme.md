# IoT-23 Training logs

These are logged prediction results of training on Aposemat IoT-23 dataset.

Evaluated on these algorithms: AdaBoost, ann, naive bayes, SVM, decision tree.

12 attributes:

Proto, duration, orig_bytes, resp_bytes, conn_state, missed_bytes, history, orig_pkts, orig_ip_bytes, resp_pkts,
resp_ip_bytes, label

using 10-folds cross-validation.

## 20-1 logs

- Train: [CTU-Malware-Capture-20-1][1]: benign 3193, malicious: 16 (99.5% / 0.5%)
- Test: [CTU-Malware-Capture-44-1][2]: benign: 211, malicious: 26 (90% / 10%)
- Test: [Honeypot-7][4]: benign: 130, malicious: 0 (100/0)

## 34-1 logs

- Train: [CTU-Malware-Capture-34-1][3]: benign 1923, malicious: 21222 (8% / 92%)
- Test: [CTU-Malware-Capture-44-1][2]: benign: 211, malicious: 26 (90% / 10%)
- Test: [Honeypot-7][4]: benign: 130, malicious: 0 (100/0)

**Decision tree**

![img](34-1-tree.png)

[1]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-20-1-labeled.csv
[2]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-44-1-labeled.csv
[3]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Malware-Capture-34-1-labeled.csv
[4]: https://github.com/iotcad/sensor-data/blob/611d9ff5e768c74fc8a5f7ea2ef52a974b85eeae/iot-23/CTU-Honeypot-Capture-7-1-labeled.csv
