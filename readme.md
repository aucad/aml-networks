# module-2

Real-Time and Edge-based Threat Detection with Anomaly Analysis

Link to notes:

<https://docs.google.com/document/d/1xqCDqrakSaXkOxijwse3zQ23Rm7-38m-rMJp7iLjS6A/edit?usp=sharing>

## Setup

Install requirements:

```text
python -m pip install -q -r requirements.txt
```

For Apple M1 chip install `tensorflow` separately:

```
python3 -m pip install tensorflow-macos
```

## Directory organization

| Directory     | Description                                          |
|:--------------|:-----------------------------------------------------|
| `adversarial` | Results of adversarial attacks on decision trees     |
| `boosted`     | Results of adversarial attacks on XGBoost classifier |
| `data`        | Preprocessed data sets for running various attacks   |
| `iot23`       | Comparison of classifiers, obtained using Weka       |
| `src/`        | Source code files                                    |
