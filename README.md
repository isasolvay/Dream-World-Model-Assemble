# Dream-World-Model-Assemble
An ensemble of various world models, majorly featuring Dreamer v2 and v3. Reimplementation of DreamerV2 model-based RL algorithm using PyTorch. The official repository for the DreamerV2 implementation can be found elsewhere.

**Please note:** This is a research project, instability and breaking changes should be expected!

## Quick Setup
To utilize dreamer V2 for Atari, execute the following command:

```python launch.py --config defaults_wis_v3 atari atari_pong dreamer_v2```

To utilize dreamer V3, execute the following command:

```python launch.py --config defaults_wis_v3 atari atari_pong dreamer_v3```

## Mlflow Tracking
This project relies considerably on Mlflow tracking to log metrics, images, store model checkpoints, and even replay buffer. It doesn't requires an installed Mlflow tracking server. By de