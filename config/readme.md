# Experiment Configurations

These experiment configuration files allow to statically configure experiment parameters.

The `default.yaml` specifies base configuration.
When running an experiment, this default file is automatically loaded.
The keys match the natural configuration options of the respective classifier, attack, etc.

This configuration can be overwritten or extended, in cascading style. For example,

- `iot.yaml` specifies configuration used only for IoT-23 experiences.
- `unsw.yaml` specifies configuration options specific to UNSW-NB15 data.

**Example of usage:**

```
python3 -m src experiment --config config/iot.yaml
```