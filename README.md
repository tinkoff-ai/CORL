# CORL (Clean Offline Reinforcement Learning)

[<img src="https://img.shields.io/badge/license-Apache_2.0-blue">](https://github.com/tinkoff-ai/CORL/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


🧵 CORL is an Offline Reinforcement Learning library that provides high-quality and easy-to-follow single-file implementations of SOTA ORL algorithms. Each implementation is backed by a research-friendly codebase, allowing you to run or tune thousands of experiments. Heavily inspired by [cleanrl](https://github.com/vwxyzjn/cleanrl) for online RL, check them out too!<br/>

* 📜 Single-file implementation
* 📈 Benchmarked Implementation for N algorithms
* 🖼 [Weights and Biases](https://wandb.ai/site) integration


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
| ✅ Behavioral Cloning <br>(BC)  |  [`any_percent_bc.py`](algorithms/any_percent_bc.py) |  [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/BC-D4RL-Results--VmlldzoyNzA2MjE1)
| ✅ Behavioral Cloning-10% <br>(BC-10%)  |  [`any_percent_bc.py`](algorithms/any_percent_bc.py) |  [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/BC-10-D4RL-Results--VmlldzoyNzEwMjcx)
| ✅ [Conservative Q-Learning for Offline Reinforcement Learning <br>(CQL)](https://arxiv.org/abs/2006.04779)  |  [`cql.py`](algorithms/cql.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/CQL-D4RL-Results--VmlldzoyNzA2MTk5)
| ✅ [Accelerating Online Reinforcement Learning with Offline Datasets <br>(AWAC)](https://arxiv.org/abs/2006.09359)  |  [`awac.py`](algorithms/awac.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/AWAC-D4RL-Results--VmlldzoyNzA2MjE3)
| ✅ [Offline Reinforcement Learning with Implicit Q-Learning <br>(IQL)](https://arxiv.org/abs/2110.06169)  |  [`iql.py`](algorithms/iql.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/IQL-D4RL-Results--VmlldzoyNzA2MTkx)
| ✅ [A Minimalist Approach to Offline Reinforcement Learning <br>(TD3+BC)](https://arxiv.org/abs/2106.06860)  |  [`td3_bc.py`](algorithms/td3_bc.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/TD3-BC-D4RL-Results--VmlldzoyNzA2MjA0)
| ✅ [Decision Transformer: Reinforcement Learning via Sequence Modeling <br>(DT)](https://arxiv.org/abs/2106.01345)  |  [`dt.py`](algorithms/dt.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/DT-D4RL-Results--VmlldzoyNzA2MTk3)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(SAC-N)](https://arxiv.org/abs/2110.01548)  |  [`sac_n.py`](algorithms/sac_n.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/SAC-N-D4RL-Results--VmlldzoyNzA1NTY1)
| ✅ [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble <br>(EDAC)](https://arxiv.org/abs/2110.01548)  |  [`edac.py`](algorithms/edac.py) | [`Gym-MuJoCo, Maze2D`](https://wandb.ai/tlab/CORL/reports/EDAC-D4RL-Results--VmlldzoyNzA5ODUw)

## D4RL Benchmarks
For learning curves and all the details, you can check the links above. Here, we report reproduced **final** and **best** scores. Note that thay differ by a big margin, and some papers may use different approaches not making it always explicit which one reporting methodology they chose.

### Last Scores
#### Gym-MuJoCo
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2 | 42.40±0.21 | 42.29±0.40 | 48.10±0.21 | 46.64±0.24 | 48.31±0.11 | 49.78±0.42 | 68.20±1.48 | 67.70±1.20 | 41.44±0.39|
|halfcheetah-medium-expert-v2 | 55.95±8.49 | 91.45±2.57 | 90.78±6.98 | 87.10±11.41 | 94.55±0.21 | 95.56±1.09 | 98.96±10.74 | 104.76±0.74 | 84.39±4.27|
|halfcheetah-medium-replay-v2 | 35.66±2.68 | 29.65±2.11 | 44.84±0.68 | 44.67±0.28 | 43.53±0.43 | 44.95±0.86 | 60.70±1.17 | 62.06±1.27 | 27.50±5.49|
|hopper-medium-v2 | 53.51±2.03 | 51.16±12.98 | 60.37±4.03 | 56.88±4.46 | 62.75±6.02 | 65.06±5.97 | 40.82±11.44 | 101.70±0.32 | 48.41±6.11|
|hopper-medium-expert-v2 | 52.30±4.63 | 105.17±7.12 | 101.17±10.48 | 86.95±17.45 | 106.24±6.09 | 105.38±7.31 | 101.31±13.43 | 105.19±11.64 | 83.20±26.68|
|hopper-medium-replay-v2 | 29.81±2.39 | 23.89±11.61 | 64.42±24.84 | 84.21±18.27 | 84.57±13.49 | 98.15±2.85 | 100.33±0.90 | 99.66±0.94 | 42.83±22.92|
|walker2d-medium-v2 | 63.23±18.76 | 58.56±4.14 | 82.71±5.51 | 80.58±3.80 | 84.03±5.42 | 69.39±31.97 | 87.47±0.76 | 93.36±1.60 | 69.15±6.76|
|walker2d-medium-expert-v2 | 98.96±18.45 | 108.45±0.30 | 110.03±0.41 | 110.23±0.48 | 111.68±0.56 | 111.65±1.74 | 114.93±0.48 | 114.75±0.86 | 92.64±3.35|
|walker2d-medium-replay-v2 | 21.80±11.72 | 41.99±17.77 | 85.62±4.63 | 82.16±2.32 | 82.55±8.00 | 80.43±3.95 | 78.99±0.58 | 87.10±3.21 | 16.93±19.57|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |      50.40 | 61.40 | 76.45 | 75.49 | 79.80 | 80.04 | 83.52 | 92.92 | 56.28 |

#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 0.36±10.03 | -2.98±6.68 | 29.41±14.22 | -6.97±17.41 | 37.69±1.99 | 60.09±19.09 | 131.08±19.36 | 90.74±6.51 | -14.55±0.15|
|maze2d-medium-v1 | 0.79±3.76 | 2.04±3.52 | 59.45±41.86 | 2.77±7.24 | 35.45±0.98 | 79.42±50.93 | 88.55±21.68 | 62.36±9.76 | -0.38±7.26|
|maze2d-large-v1 | 2.26±5.07 | 3.14±4.77 | 97.10±29.34 | 1.29±7.11 | 49.64±22.02 | 217.44±4.93 | 205.13±1.33 | 108.17±25.02 | -0.45±1.51|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 1.13 | 0.74 | 61.99 | -0.97 | 40.92 | 118.98 | 141.59 | 87.09 | -5.13 |

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 51.50±8.81 | 0.00±0.00 | 93.25±1.50 | 63.75±8.26 | 74.50±11.03 | 63.50±9.33 | 0.00±0.00 | 29.25±33.35 | 52.75±11.47|
|antmaze-medium-play-v0 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 71.50±12.56 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|antmaze-large-play-v0 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 40.75±12.69 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 17.17 | 0.00 | 31.08 | 21.25 | 62.25 | 21.17 | 0.00 | 9.75 | 17.58 |

### Best Scores
#### Gym-MuJoCo
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2 | 43.60±0.16 | 43.74±0.18 | 48.93±0.13 | 47.26±0.23 | 48.77±0.06 | 50.79±0.19 | 72.21±0.35 | 69.72±1.06 | 42.63±0.09|
|halfcheetah-medium-expert-v2 | 79.69±3.58 | 93.98±0.18 | 96.59±1.01 | 95.82±0.31 | 95.83±0.38 | 96.85±0.32 | 111.73±0.55 | 110.62±1.20 | 87.34±0.65|
|halfcheetah-medium-replay-v2 | 40.52±0.22 | 41.45±0.10 | 45.84±0.30 | 45.97±0.32 | 45.06±0.16 | 46.56±0.27 | 67.29±0.39 | 66.55±1.21 | 32.20±2.50|
|hopper-medium-v2 | 69.04±3.35 | 66.91±2.30 | 70.44±1.37 | 69.09±0.85 | 80.74±1.27 | 99.25±0.87 | 101.79±0.23 | 103.26±0.16 | 61.95±4.63|
|hopper-medium-expert-v2 | 90.63±12.68 | 113.05±0.17 | 113.22±0.50 | 111.01±1.93 | 111.79±0.47 | 113.25±0.50 | 111.24±0.17 | 111.80±0.13 | 107.01±3.28|
|hopper-medium-replay-v2 | 68.88±11.93 | 53.82±8.10 | 98.12±1.34 | 102.10±0.41 | 102.33±0.44 | 101.68±0.38 | 103.83±0.61 | 103.28±0.57 | 59.65±13.50|
|walker2d-medium-v2 | 80.64±1.06 | 80.46±1.41 | 86.91±0.32 | 84.76±0.15 | 87.99±0.83 | 85.98±4.43 | 90.17±0.63 | 95.78±1.23 | 75.54±0.53|
|walker2d-medium-expert-v2 | 109.95±0.72 | 109.57±0.33 | 112.21±0.07 | 111.70±0.28 | 113.19±0.33 | 113.30±2.51 | 116.93±0.49 | 116.52±0.86 | 96.30±1.18|
|walker2d-medium-replay-v2 | 48.41±8.78 | 71.54±1.16 | 91.17±0.83 | 88.02±1.18 | 91.85±2.26 | 86.79±0.96 | 85.18±1.89 | 89.69±1.60 | 67.23±6.73|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |     70.15 | 74.95 | 84.83 | 83.97 | 86.40 | 88.27 | 95.60 | 96.36 | 69.98 |


#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 16.09±1.00 | 16.85±0.60 | 99.33±18.66 | 18.82±0.63 | 44.04±3.02 | 137.96±12.50 | 151.28±8.14 | 144.30±5.60 | -14.19±0.56|
|maze2d-medium-v1 | 19.16±1.44 | 24.81±4.09 | 150.93±4.50 | 17.96±5.24 | 92.25±40.74 | 152.11±23.00 | 90.04±20.74 | 150.82±2.76 | 45.13±6.25|
|maze2d-large-v1 | 20.75±7.69 | 35.66±6.40 | 197.64±6.07 | 12.27±5.34 | 138.70±44.70 | 227.79±1.99 | 207.10±1.46 | 179.90±2.41 | 3.94±2.24|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 18.67 | 25.77 | 149.30 | 16.35 | 91.66 | 172.62 | 149.47 | 158.34 | 11.63 |

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 71.25±9.07 | 0.00±0.00 | 97.75±1.50 | 76.75±4.99 | 87.00±2.94 | 74.75±8.77 | 0.00±0.00 | 75.00±27.51 | 65.50±8.96|
|antmaze-medium-play-v0 | 4.75±2.22 | 0.00±0.00 | 6.00±2.00 | 1.75±0.96 | 86.00±2.16 | 14.00±11.80 | 0.00±0.00 | 0.00±0.00 | 1.00±2.00|
|antmaze-large-play-v0 | 0.75±0.50 | 0.00±0.00 | 0.50±0.58 | 0.00±0.00 | 53.00±6.83 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 25.58 | 0.00 | 34.75 | 26.17 | 75.33 | 29.58 | 0.00 | 25.00 | 22.17 |

## Citing CORL

If you use CORL in your work, please use the following bibtex
```bibtex
@misc{corl2022,
  author={Tarasov, Denis and Nikulin, Alexander and Akimov, Dmitriy and Kurenkov, Vladislav and Sergey Kolesnikov},
  title={CORL: Research-oriented Deep Offline Reinforcement Learning Library},
  year={2022},
  url={https://github.com/tinkoff-ai/CORL},
}
```
