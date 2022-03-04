# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Get dependencies:
```
pip install -r requirements.txt
```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```

## Benchmarks
So far, the following benchmarks can be used for testing.
| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [Atari 100k](https://github.com/openai/atari-py) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilitie