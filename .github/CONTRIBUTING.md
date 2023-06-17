## Native Execution

These steps explain how to run experiments from source natively on host machine.
You should also follow these steps, if you want to prepare a development environment and make code changes.

This implementation is not compatible with Apple M1 machines due to underlying dependency (tensorflow-macos); and 
although it does not prevent most experiments, some issues may surface periodically.
A machine with AMD64 architecture is recommended for high fidelity.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it [including the submodule](https://stackoverflow.com/a/4438292) typically:

  ```
  git clone --recurse-submodules https://github.com/iotcad/aml-networks.git
  ```
  
**Step 1: Build robust XGBoost**

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies and **must be built
locally from source** (the submodule `RobustTrees`).

By default, you will need gcc compiler with OpenMP support. Assuming a suitable environment, run:

```
cd RobustTrees
make -j4
```

If the build causes issues, follow [these instructions](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source)
to build it from source, and explore the various build options for different environments.


**Step 2: Install dependencies**

Install required Python dependencies.

```
python3 -m pip install -r requirements-dev.txt
```

Install XGBoost from the local build location:

```
python3 -m pip install -e "/path/to/RobustTrees/python-package"
```

After setup, check the xgboost runtime:

```
python3 -m pip show xgboost
```

The version number should be 0.72. 

Next, run:

```
python3 -m aml
```

This should produce a help prompt with instructions.
You are now ready to run experiments and make code changes. 
