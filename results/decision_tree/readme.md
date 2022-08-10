# Adversarial attacks on decision tree classifier

We apply adversarial attacks on a (single) non-robust decision tree, trained using IoT-23 dataset:
CTU-Malware-Capture-44-1, with 237 rows (benign 211, malicious 26), 90 / 10 split; trained using basic
[scikit learn decision tree][1]. We then apply various adversarial attacks from the [Adversarial Robustness Toolbox][2]
to attack this classifier.

## Results

### Attribute inference attack

The attacked feature must be categorical, and with a relatively small number of possible values (preferably binary, but
should at least be less then the number of label classes). 
[[implementation]](https://github.com/iotcad/module-2/blob/bc906db0fda265ea4928f9d0899b3e00bb17a928/src/attack_inf.py)

Output of black-box inference on attributes `proto=udp` and `conn_state=SF`

```text
Read dataset ----------------- data/CTU-44-1.csv
Attributes ------------------- 22
Classes ---------------------- malicious, benign
Training instances ----------- 189
Test instances --------------- 48
Test split ------------------- 3/45
Accuracy --------------------- 100.00 %
Precision -------------------- 100.00 %
Recall ----------------------- 100.00 %
F-score ---------------------- 100.00 %
* Inference of attribute proto=udp:
Baseline attack -------------- Accuracy: 98.95 % Precision: 100.00 % Recall: 90.91 %
Black box attack ------------- Accuracy: 98.95 % Precision: 100.00 % Recall: 90.91 %
White box 1 attack ----------- Accuracy: 97.89 % Precision: 90.91 % Recall: 90.91 %
White box 2 attack ----------- Accuracy: 87.37 % Precision: 0.00 % Recall: 0.00 %
Membership attack ------------ Accuracy: 11.58 % Precision: 11.58 % Recall: 100.00 %
* Inference of attribute conn_state=SF:
Baseline attack -------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
Black box attack ------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
White box 1 attack ----------- Accuracy: 61.05 % Precision: 100.00 % Recall: 0.00 %
White box 2 attack ----------- Accuracy: 61.05 % Precision: 100.00 % Recall: 0.00 %
Membership attack ------------ Accuracy: 38.95 % Precision: 38.95 % Recall: 100.00 %
* Inference of attribute conn_state=S0:
Baseline attack -------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
Black box attack ------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
White box 1 attack ----------- Accuracy: 61.05 % Precision: 61.05 % Recall: 100.00 %
White box 2 attack ----------- Accuracy: 61.05 % Precision: 61.05 % Recall: 100.00 %
Membership attack ------------ Accuracy: 61.05 % Precision: 61.05 % Recall: 100.00 %
* Inference of attribute history=S:
Baseline attack -------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
Black box attack ------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
White box 1 attack ----------- Accuracy: 92.63 % Precision: 92.63 % Recall: 100.00 %
White box 2 attack ----------- Accuracy: 92.63 % Precision: 92.63 % Recall: 100.00 %
Membership attack ------------ Accuracy: 92.63 % Precision: 92.63 % Recall: 100.00 %
* Inference of attribute history=Dd:
Baseline attack -------------- Accuracy: 98.95 % Precision: 100.00 % Recall: 97.56 %
Black box attack ------------- Accuracy: 98.95 % Precision: 100.00 % Recall: 97.56 %
White box 1 attack ----------- Accuracy: 56.84 % Precision: 100.00 % Recall: 0.00 %
White box 2 attack ----------- Accuracy: 56.84 % Precision: 100.00 % Recall: 0.00 %
Membership attack ------------ Accuracy: 43.16 % Precision: 43.16 % Recall: 100.00 %
* Inference of attribute history=D:
Baseline attack -------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
Black box attack ------------- Accuracy: 100.00 % Precision: 100.00 % Recall: 100.00 %
White box 1 attack ----------- Accuracy: 68.42 % Precision: 68.42 % Recall: 100.00 %
White box 2 attack ----------- Accuracy: 68.42 % Precision: 68.42 % Recall: 100.00 %
Membership attack ------------ Accuracy: 68.42 % Precision: 68.42 % Recall: 100.00 %
```

**Inference attack methods**

- [**Baseline attack**][BL]: Implementation of a baseline attribute inference, not using a model. The idea is to 
    train a simple neural network to learn the attacked feature from the rest of the features. Should be used to 
    compare with other attribute inference results. 

- [**Black box attack**][BB]: Implementation of a simple black-box attribute inference attack. The idea is to train 
    a simple neural network to learn the attacked feature from the rest of the features and the model’s predictions. 
    Assumes the availability of the attacked model’s predictions for the samples under attack, in addition to the rest 
    of the feature values. If this is not available, the true class label of the samples may be used as a proxy. 

- [**White-box 1 attack**][W1]: A variation of the method proposed by of Fredrikson et al. Assumes the availability of 
    the attacked model’s predictions for the samples under attack, in addition to access to the model itself and the 
    rest of the feature values. If this is not available, the true class label of the samples may be used as a proxy. 
    Also assumes that the attacked feature is discrete or categorical, with limited number of possible values, for 
    example: a boolean feature. Paper link: <https://dl.acm.org/doi/10.1145/2810103.2813677> [^1]

- [**White-box 2 attack**][W2]: Implementation of Fredrikson et al. white box inference attack for decision trees. 
    Assumes that the attacked feature is discrete or categorical, with limited number of possible values, for 
    example: a boolean feature. Paper link: <https://dl.acm.org/doi/10.1145/2810103.2813677> [^1]
     
- [**Membership attack**][MS]: Implementation of an attribute inference attack that utilizes a membership inference 
    attack. The idea is to find the target feature value that causes the membership inference attack to classify the 
    sample as a member with the highest confidence.


### ZOO Evasion attack

Applying Zeroth-Order Optimization (ZOO) Attack, on CTU-44-1 dataset, applied to decision tree classifier:

- blue circles: malicious training 
- green circles: benign training 
- red crosses: adversarial modified instance
- black line: difference between original and adversarial

 
![img](non_robust_1.png) 
 
![img](non_robust_2.png) 
  
![img](non_robust_3.png) 
 
![img](non_robust_4.png)

  
<!-- references -->
  
[1]: https://scikit-learn.org/stable/modules/tree.html  
[2]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/

[BL]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-baseline
[BB]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-black-box
[W1]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-white-box-decision-tree
[W2]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-white-box-lifestyle-decision-tree
[MS]: https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-membership
