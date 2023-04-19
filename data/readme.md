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

### Preprocessing instructions

Same steps apply to both IoT-23 and UNSW-NB15 data. These steps are easy to do with [Weka](https://waikato.github.io/weka-wiki/downloading_weka/).

1. Remove irrelevant features. In other words, _keep_ the following:  
   
   ```
   FOR IOT-23 -- KEEP:   
   proto, duration, orig_bytes, resp_bytes, conn_state,  missed_bytes,  
   history, orig_pkts,  orig_ip_bytes,  resp_pkts,  resp_ip_bytes, label

   FOR UNSW-NB15 -- KEEP:
   proto, state, service , dur, sbytes, dbytes, sloss, dloss, swin, dwin, stcpb, dtcpb, smeansz,
   dmeansz, sjit, djit, trans_depth, ct_srv_src, ct_srv_dst, ct_src_ ltm, ct_dst_ltm, 
   ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, label
   ```

2. For each remaining nominal attribute, check if they have many values, e.g., `history` for IoT-23.
   Merge low-frequency values. After this, at most 2 or 3 distinct values should remain for nominal attributes.

3. One hot encode nominal attributes (in Weka it is called "Nominal to Binary" filter)

4. Change class labels to numeric values: Benign = 0 and Malicious = 1.

5. Missing values should be actual nulls (not `-` `?` etc.).

6. Use Weka's supervised instance `SpreadSubsample` with `distributionSpread=1.0`, 
   to obtain a random sample of desired size, with equal class distribution (see [details here](https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/)).

7. Save as a comma-separated file; the expects inputs is csv-format.