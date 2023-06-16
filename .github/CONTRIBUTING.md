## Native Execution

These steps explain how to run experiments from source natively on host machine.
You should also follow these steps, if you want to prepare a development environment and make code changes.

- :snake: **Required** Python environment: 3.8 or 3.9

- :warning: **Submodule** This repository has a submodule. Clone it, including the submodule
  [(instructions)](https://stackoverflow.com/a/4438292).

This implementation is not compatible with Apple M1 machines due to underlying dependency (tensorflow-macos); and 
although it does not prevent most experiments, some issues may arise sporadically.
A machine with AMD64/x86 architecture is recommended.

**Step 1: Build robust XGBoost**

The evaluation uses a modified version of XGBoost classifier, enhanced with adversarial robustness property. This
classifier is not installed with the other package dependencies. The XGBoost classifier must be built
locally [from source](https://github.com/chenhongge/RobustTrees) (also included as a submodule `RobustTrees`).
Follow the [instructions here](https://github.com/chenhongge/RobustTrees/tree/master/python-package#from-source)
to build it from source.

**Step 2: Install dependencies**

Install required Python dependencies.

```
python3 -m pip install -r requirements-dev.txt
```

Install XGBoost from the local build location:

```
python3 -m pip install -e "/path/to/RobustTrees/python-package"
```

After setup, check the runtime:

```
python3 -m pip show xgboost
```

The version number should be 0.72. You are now ready to run experiments and make code changes. 

To check changes with a linter, run:

```
make lint
```
