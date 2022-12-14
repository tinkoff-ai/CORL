# CORL (Clean Offline Reinforcement Learning)

[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


๐งต CORL is an Offline Reinforcement Learning library that provides high-quality and easy-to-follow single-file implementations of SOTA ORL algorithms. Each implementation is backed by a research-friendly codebase, allowing you to run or tune thousands of experiments. Heavily inspired by [cleanrl](https://github.com/vwxyzjn/cleanrl) for online RL, check them out too!<br/>

* ๐ Single-file implementation
* ๐ Benchmarked Implementation for N algorithms
* ๐ผ [Weights and Biases](https://wandb.ai/site) integration


## Getting started

```bash
git clone https://github.com/tinkoff-ai/CORL.git && cd CORL
pip install -r requirements/requirements_dev.txt

# alternatively, you could use docker
docker build -t <image_name> .
docker run gpus=all -it --rm --name <container_name> <image_name>
```


## Algorithms Implemented

| Algorithm      | Variants Implemented | Wandb Report |
| ----------- | ----------- | ----------- |
| โ Behavioral Cloning <br>(BC)  |  [`any_percent_bc.py`](algorithms/any_percent_bc.py) |  [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/BC-D4RL-Results--VmlldzoyNzA2MjE1)
| โ Behavioral Cloning-10% <br>(BC-10%)  |  [`any_percent_bc.py`](algorithms/any_percent_bc.py) |  [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/BC-10-D4RL-Results--VmlldzoyNzEwMjcx)
| โ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)  |  [`cql.py`](algorithms/cql.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/CQL-D4RL-Results--VmlldzoyNzA2MTk5)
| โ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)  |  [`awac.py`](algorithms/awac.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/AWAC-D4RL-Results--VmlldzoyNzA2MjE3)
| โ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)  |  [`iql.py`](algorithms/iql.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/IQL-D4RL-Results--VmlldzoyNzA2MTkx)
| โ [A Minimalist Approach to Offline Reinforcement Learning <br>(TD3+BC)](https://arxiv.org/abs/2106.06860)  |  [`td3_bc.py`](algorithms/td3_bc.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/TD3-BC-D4RL-Results--VmlldzoyNzA2MjA0)
| โ [Decision Transformer: Reinforcement Learning via Sequence Modeling <br>(DT)](https://arxiv.org/abs/2106.01345)  |  [`dt.py`](algorithms/dt.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/DT-D4RL-Results--VmlldzoyNzA2MTk3)
| โ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(SAC-N)](https://arxiv.org/abs/2110.01548)  |  [`sac_n.py`](algorithms/sac_n.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/SAC-N-D4RL-Results--VmlldzoyNzA1NTY1)
| โ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(EDAC)](https://arxiv.org/abs/2110.01548)  |  [`edac.py`](algorithms/edac.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/EDAC-D4RL-Results--VmlldzoyNzA5ODUw)

## D4RL Benchmarks
For learning curves and all the details, you can check the links above. Here, we report reproduced **final** and **best** scores. Note that thay differ by a big margin, and some papers may use different approaches not making it always explicit which one reporting methodology they chose.

### Last Scores
#### Gym-MuJoCo
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2 | 42.40ยฑ0.21 | 42.46ยฑ0.81 | 48.10ยฑ0.21 | 47.08ยฑ0.19 | 48.31ยฑ0.11 | 50.01ยฑ0.30 | 68.20ยฑ1.48 | 67.70ยฑ1.20 | 42.20ยฑ0.30|
|halfcheetah-medium-expert-v2 | 55.95ยฑ8.49 | 90.10ยฑ2.83 | 90.78ยฑ6.98 | 95.98ยฑ0.83 | 94.55ยฑ0.21 | 95.29ยฑ0.91 | 98.96ยฑ10.74 | 104.76ยฑ0.74 | 91.55ยฑ1.10|
|halfcheetah-medium-replay-v2 | 35.66ยฑ2.68 | 23.59ยฑ8.02 | 44.84ยฑ0.68 | 45.19ยฑ0.58 | 43.53ยฑ0.43 | 44.91ยฑ1.30 | 60.70ยฑ1.17 | 62.06ยฑ1.27 | 38.91ยฑ0.57|
|hopper-medium-v2 | 53.51ยฑ2.03 | 55.48ยฑ8.43 | 60.37ยฑ4.03 | 64.98ยฑ6.12 | 62.75ยฑ6.02 | 63.69ยฑ4.29 | 40.82ยฑ11.44 | 101.70ยฑ0.32 | 65.10ยฑ1.86|
|hopper-medium-expert-v2 | 52.30ยฑ4.63 | 111.16ยฑ1.19 | 101.17ยฑ10.48 | 93.89ยฑ14.34 | 106.24ยฑ6.09 | 105.29ยฑ7.19 | 101.31ยฑ13.43 | 105.19ยฑ11.64 | 110.44ยฑ0.39|
|hopper-medium-replay-v2 | 29.81ยฑ2.39 | 70.42ยฑ9.99 | 64.42ยฑ24.84 | 87.67ยฑ14.42 | 84.57ยฑ13.49 | 98.15ยฑ2.85 | 100.33ยฑ0.90 | 99.66ยฑ0.94 | 81.77ยฑ7.93|
|walker2d-medium-v2 | 63.23ยฑ18.76 | 67.34ยฑ5.97 | 82.71ยฑ5.51 | 80.38ยฑ3.45 | 84.03ยฑ5.42 | 69.39ยฑ31.97 | 87.47ยฑ0.76 | 93.36ยฑ1.60 | 67.63ยฑ2.93|
|walker2d-medium-expert-v2 | 98.96ยฑ18.45 | 108.70ยฑ0.29 | 110.03ยฑ0.41 | 109.68ยฑ0.52 | 111.68ยฑ0.56 | 111.16ยฑ2.41 | 114.93ยฑ0.48 | 114.75ยฑ0.86 | 107.11ยฑ1.11|
|walker2d-medium-replay-v2 | 21.80ยฑ11.72 | 54.35ยฑ7.32 | 85.62ยฑ4.63 | 79.24ยฑ4.97 | 82.55ยฑ8.00 | 71.73ยฑ13.98 | 78.99ยฑ0.58 | 87.10ยฑ3.21 | 59.86ยฑ3.15|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |50.40 | 69.29 | 76.45 | 78.23 | 79.80 | 78.85 | 83.52 | 92.92 | 73.84|

#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 0.36ยฑ10.03 | 12.18ยฑ4.95 | 29.41ยฑ14.22 | -14.83ยฑ0.47 | 37.69ยฑ1.99 | 68.30ยฑ25.72 | 130.59ยฑ19.08 | 95.26ยฑ7.37 | 18.08ยฑ29.35|
|maze2d-medium-v1 | 0.79ยฑ3.76 | 14.25ยฑ2.69 | 59.45ยฑ41.86 | 86.62ยฑ11.11 | 35.45ยฑ0.98 | 82.66ยฑ46.71 | 88.61ยฑ21.62 | 57.04ยฑ3.98 | 31.71ยฑ30.40|
|maze2d-large-v1 | 2.26ยฑ5.07 | 11.32ยฑ5.88 | 97.10ยฑ29.34 | 33.22ยฑ43.66 | 49.64ยฑ22.02 | 218.87ยฑ3.96 | 204.76ยฑ1.37 | 95.60ยฑ26.46 | 35.66ยฑ32.56|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 1.13 | 12.58 | 61.99 | 35.00 | 40.92 | 123.28 | 141.32 | 82.64 | 28.48|

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 51.50ยฑ8.81 | 67.75ยฑ6.40 | 93.25ยฑ1.50 | 72.75ยฑ5.32 | 74.50ยฑ11.03 | 63.50ยฑ9.33 | 0.00ยฑ0.00 | 29.25ยฑ33.35 | 51.75ยฑ11.76|
|antmaze-medium-play-v0 | 0.00ยฑ0.00 | 2.50ยฑ1.91 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 71.50ยฑ12.56 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00|
|antmaze-large-play-v0 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 40.75ยฑ12.69 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 17.17 | 23.42 | 31.08 | 24.25 | 62.25 | 21.17 | 0.00 | 9.75 | 17.25 |

### Best Scores
#### Gym-MuJoCo
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2 | 43.60ยฑ0.16 | 43.90ยฑ0.15 | 48.93ยฑ0.13 | 47.45ยฑ0.10 | 48.77ยฑ0.06 | 50.87ยฑ0.21 | 72.21ยฑ0.35 | 69.72ยฑ1.06 | 42.73ยฑ0.11|
|halfcheetah-medium-expert-v2 | 79.69ยฑ3.58 | 94.11ยฑ0.25 | 96.59ยฑ1.01 | 96.74ยฑ0.14 | 95.83ยฑ0.38 | 96.87ยฑ0.31 | 111.73ยฑ0.55 | 110.62ยฑ1.20 | 93.40ยฑ0.25|
|halfcheetah-medium-replay-v2 | 40.52ยฑ0.22 | 42.27ยฑ0.53 | 45.84ยฑ0.30 | 46.38ยฑ0.14 | 45.06ยฑ0.16 | 46.57ยฑ0.27 | 67.29ยฑ0.39 | 66.55ยฑ1.21 | 40.31ยฑ0.32|
|hopper-medium-v2 | 69.04ยฑ3.35 | 73.84ยฑ0.43 | 70.44ยฑ1.37 | 77.47ยฑ6.00 | 80.74ยฑ1.27 | 99.40ยฑ1.12 | 101.79ยฑ0.23 | 103.26ยฑ0.16 | 69.42ยฑ4.21|
|hopper-medium-expert-v2 | 90.63ยฑ12.68 | 113.13ยฑ0.19 | 113.22ยฑ0.50 | 112.74ยฑ0.07 | 111.79ยฑ0.47 | 113.37ยฑ0.63 | 111.24ยฑ0.17 | 111.80ยฑ0.13 | 111.18ยฑ0.24|
|hopper-medium-replay-v2 | 68.88ยฑ11.93 | 90.57ยฑ2.38 | 98.12ยฑ1.34 | 102.20ยฑ0.38 | 102.33ยฑ0.44 | 101.76ยฑ0.43 | 103.83ยฑ0.61 | 103.28ยฑ0.57 | 88.74ยฑ3.49|
|walker2d-medium-v2 | 80.64ยฑ1.06 | 82.05ยฑ1.08 | 86.91ยฑ0.32 | 84.57ยฑ0.15 | 87.99ยฑ0.83 | 86.22ยฑ4.58 | 90.17ยฑ0.63 | 95.78ยฑ1.23 | 74.70ยฑ0.64|
|walker2d-medium-expert-v2 | 109.95ยฑ0.72 | 109.90ยฑ0.10 | 112.21ยฑ0.07 | 111.63ยฑ0.20 | 113.19ยฑ0.33 | 113.40ยฑ2.57 | 116.93ยฑ0.49 | 116.52ยฑ0.86 | 108.71ยฑ0.39|
|walker2d-medium-replay-v2 | 48.41ยฑ8.78 | 76.09ยฑ0.47 | 91.17ยฑ0.83 | 89.34ยฑ0.59 | 91.85ยฑ2.26 | 87.06ยฑ0.93 | 85.18ยฑ1.89 | 89.69ยฑ1.60 | 68.22ยฑ1.39|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |    70.15 | 80.65 | 84.83 | 85.39 | 86.40 | 88.39 | 95.60 | 96.36 | 77.49 |


#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 16.09ยฑ1.00 | 22.49ยฑ1.75 | 99.33ยฑ18.66 | 84.92ยฑ34.40 | 44.04ยฑ3.02 | 141.92ยฑ12.88 | 153.12ยฑ7.50 | 149.88ยฑ2.27 | 63.83ยฑ20.04|
|maze2d-medium-v1 | 19.16ยฑ1.44 | 27.64ยฑ2.16 | 150.93ยฑ4.50 | 137.52ยฑ9.83 | 92.25ยฑ40.74 | 160.95ยฑ11.64 | 93.80ยฑ16.93 | 154.41ยฑ1.82 | 68.14ยฑ14.15|
|maze2d-large-v1 | 20.75ยฑ7.69 | 41.83ยฑ4.20 | 197.64ยฑ6.07 | 153.29ยฑ12.86 | 138.70ยฑ44.70 | 228.00ยฑ2.06 | 207.51ยฑ1.11 | 182.52ยฑ3.10 | 50.25ยฑ22.33|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 18.67 | 30.65 | 149.30 | 125.25 | 91.66 | 176.96 | 151.48 | 162.27 | 60.74 |

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 71.25ยฑ9.07 | 79.50ยฑ2.38 | 97.75ยฑ1.50 | 85.00ยฑ3.56 | 87.00ยฑ2.94 | 74.75ยฑ8.77 | 0.00ยฑ0.00 | 75.00ยฑ27.51 | 60.50ยฑ3.11|
|antmaze-medium-play-v0 | 4.75ยฑ2.22 | 8.50ยฑ3.51 | 6.00ยฑ2.00 | 3.00ยฑ0.82 | 86.00ยฑ2.16 | 14.00ยฑ11.80 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.25ยฑ0.50|
|antmaze-large-play-v0 | 0.75ยฑ0.50 | 11.75ยฑ2.22 | 0.50ยฑ0.58 | 0.50ยฑ0.58 | 53.00ยฑ6.83 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00 | 0.00ยฑ0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 25.58 | 33.25 | 34.75 | 29.50 | 75.33 | 29.58 | 0.00 | 25.00 | 20.25 |

## Citing CORL

If you use CORL in your work, please use the following bibtex
```bibtex
@inproceedings{
tarasov2022corl,
  title={{CORL}: Research-oriented Deep Offline Reinforcement Learning Library},
  author={Denis Tarasov and Alexander Nikulin and Dmitry Akimov and Vladislav Kurenkov and Sergey Kolesnikov},
  booktitle={3rd Offline RL Workshop: Offline RL as a ''Launchpad''},
  year={2022},
  url={https://openreview.net/forum?id=SyAS49bBcv}
}
```
