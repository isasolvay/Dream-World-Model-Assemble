# Dream-World-Model-Assemble
An ensemble of various world models, majorly featuring Dreamer v2 and v3. Reimplementation of DreamerV2 model-based RL algorithm using PyTorch. The official repository for the DreamerV2 implementation can be found elsewhere.

**Please note:** This is a research project, instability and breaking changes should be expected!

## Quick Setup
To utilize dreamer V2 for Atari, execute the following command:

```python launch.py --config defaults_wis_v3 atari atari_pong dreamer_v2```

To utilize dreamer V3, execute the following command:

```python launch.py --config defaults_wis_v3 atari atari_pong dreamer_v3```

## Mlflow Tracking
This project relies considerably on Mlflow tracking to log metrics, images, store model checkpoints, and even replay buffer. It doesn't requires an installed Mlflow tracking server. By default, mlflow is just a pip package, storing all metrics and files locally under `./mlruns` directory.

However, if you're running experiments on the cloud, it would be convenient to set up a persistent Mlflow tracking server. In this case, you just need to set the `MLFLOW_TRACKING_URI` env variable, and all the metrics will be redirected to the server instead of the local directory.

Please be aware that the replay buffer is just a directory with mlflow artifacts in `*.npz` format. Hence, if you set up an S3 or GCS mlflow artifact store,