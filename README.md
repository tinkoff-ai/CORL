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
|halfcheetah-medium-v2 | 42.40±0.21 | 42.46±0.81 | 48.10±0.21 | 46.64±0.24 | 48.31±0.11 | 50.01±0.30 | 68.20±1.48 | 67.70±1.20 | 42.20±0.30|
|halfcheetah-medium-expert-v2 | 55.95±8.49 | 90.10±2.83 | 90.78±6.98 | 87.10±11.41 | 94.55±0.21 | 95.29±0.91 | 98.96±10.74 | 104.76±0.74 | 91.55±1.10|
|halfcheetah-medium-replay-v2 | 35.66±2.68 | 23.59±8.02 | 44.84±0.68 | 44.67±0.28 | 43.53±0.43 | 44.91±1.30 | 60.70±1.17 | 62.06±1.27 | 38.91±0.57|
|hopper-medium-v2 | 53.51±2.03 | 55.48±8.43 | 60.37±4.03 | 56.88±4.46 | 62.75±6.02 | 63.69±4.29 | 40.82±11.44 | 101.70±0.32 | 65.10±1.86|
|hopper-medium-expert-v2 | 52.30±4.63 | 111.16±1.19 | 101.17±10.48 | 86.95±17.45 | 106.24±6.09 | 105.29±7.19 | 101.31±13.43 | 105.19±11.64 | 110.44±0.39|
|hopper-medium-replay-v2 | 29.81±2.39 | 70.42±9.99 | 64.42±24.84 | 84.21±18.27 | 84.57±13.49 | 98.15±2.85 | 100.33±0.90 | 99.66±0.94 | 81.77±7.93|
|walker2d-medium-v2 | 63.23±18.76 | 67.34±5.97 | 82.71±5.51 | 80.58±3.80 | 84.03±5.42 | 69.39±31.97 | 87.47±0.76 | 93.36±1.60 | 67.63±2.93|
|walker2d-medium-expert-v2 | 98.96±18.45 | 108.70±0.29 | 110.03±0.41 | 110.23±0.48 | 111.68±0.56 | 111.16±2.41 | 114.93±0.48 | 114.75±0.86 | 107.11±1.11|
|walker2d-medium-replay-v2 | 21.80±11.72 | 54.35±7.32 | 85.62±4.63 | 82.16±2.32 | 82.55±8.00 | 71.73±13.98 | 78.99±0.58 | 87.10±3.21 | 59.86±3.15|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |50.40 | 69.29 | 76.45 | 75.49 | 79.80 | 78.85 | 83.52 | 92.92 | 73.84 |

#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 0.36±10.03 | 12.18±4.95 | 29.41±14.22 | -6.97±17.41 | 37.69±1.99 | 68.30±25.72 | 130.59±19.08 | 95.26±7.37 | 18.08±29.35|
|maze2d-medium-v1 | 0.79±3.76 | 14.25±2.69 | 59.45±41.86 | 2.77±7.24 | 35.45±0.98 | 82.66±46.71 | 88.61±21.62 | 57.04±3.98 | 31.71±30.40|
|maze2d-large-v1 | 2.26±5.07 | 11.32±5.88 | 97.10±29.34 | 1.29±7.11 | 49.64±22.02 | 218.87±3.96 | 204.76±1.37 | 95.60±26.46 | 35.66±32.56|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 1.13 | 12.58 | 61.99 | -0.97 | 40.92 | 123.28 | 141.32 | 82.64 | 28.48 |

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 51.50±8.81 | 67.75±6.40 | 93.25±1.50 | 63.75±8.26 | 74.50±11.03 | 63.50±9.33 | 0.00±0.00 | 29.25±33.35 | 51.75±11.76|
|antmaze-medium-play-v0 | 0.00±0.00 | 2.50±1.91 | 0.00±0.00 | 0.00±0.00 | 71.50±12.56 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|antmaze-large-play-v0 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 40.75±12.69 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 17.17 | 23.42 | 31.08 | 21.25 | 62.25 | 21.17 | 0.00 | 9.75 | 17.25 |

### Best Scores
#### Gym-MuJoCo
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|halfcheetah-medium-v2 | 43.60±0.16 | 43.90±0.15 | 48.93±0.13 | 47.26±0.23 | 48.77±0.06 | 50.87±0.21 | 72.21±0.35 | 69.72±1.06 | 42.73±0.11|
|halfcheetah-medium-expert-v2 | 79.69±3.58 | 94.11±0.25 | 96.59±1.01 | 95.82±0.31 | 95.83±0.38 | 96.87±0.31 | 111.73±0.55 | 110.62±1.20 | 93.40±0.25|
|halfcheetah-medium-replay-v2 | 40.52±0.22 | 42.27±0.53 | 45.84±0.30 | 45.97±0.32 | 45.06±0.16 | 46.57±0.27 | 67.29±0.39 | 66.55±1.21 | 40.31±0.32|
|hopper-medium-v2 | 69.04±3.35 | 73.84±0.43 | 70.44±1.37 | 69.09±0.85 | 80.74±1.27 | 99.40±1.12 | 101.79±0.23 | 103.26±0.16 | 69.42±4.21|
|hopper-medium-expert-v2 | 90.63±12.68 | 113.13±0.19 | 113.22±0.50 | 111.01±1.93 | 111.79±0.47 | 113.37±0.63 | 111.24±0.17 | 111.80±0.13 | 111.18±0.24|
|hopper-medium-replay-v2 | 68.88±11.93 | 90.57±2.38 | 98.12±1.34 | 102.10±0.41 | 102.33±0.44 | 101.76±0.43 | 103.83±0.61 | 103.28±0.57 | 88.74±3.49|
|walker2d-medium-v2 | 80.64±1.06 | 82.05±1.08 | 86.91±0.32 | 84.76±0.15 | 87.99±0.83 | 86.22±4.58 | 90.17±0.63 | 95.78±1.23 | 74.70±0.64|
|walker2d-medium-expert-v2 | 109.95±0.72 | 109.90±0.10 | 112.21±0.07 | 111.70±0.28 | 113.19±0.33 | 113.40±2.57 | 116.93±0.49 | 116.52±0.86 | 108.71±0.39|
|walker2d-medium-replay-v2 | 48.41±8.78 | 76.09±0.47 | 91.17±0.83 | 88.02±1.18 | 91.85±2.26 | 87.06±0.93 | 85.18±1.89 | 89.69±1.60 | 68.22±1.39|
|                              |            |        |        |     |     |      |       |      |    |
| **locomotion average**       |    70.15 | 80.65 | 84.83 | 83.97 | 86.40 | 88.39 | 95.60 | 96.36 | 77.49 |


#### Maze2d
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|maze2d-umaze-v1 | 16.09±1.00 | 22.49±1.75 | 99.33±18.66 | 18.82±0.63 | 44.04±3.02 | 141.92±12.88 | 153.12±7.50 | 149.88±2.27 | 63.83±20.04|
|maze2d-medium-v1 | 19.16±1.44 | 27.64±2.16 | 150.93±4.50 | 17.96±5.24 | 92.25±40.74 | 160.95±11.64 | 93.80±16.93 | 154.41±1.82 | 68.14±14.15|
|maze2d-large-v1 | 20.75±7.69 | 41.83±4.20 | 197.64±6.07 | 12.27±5.34 | 138.70±44.70 | 228.00±2.06 | 207.51±1.11 | 182.52±3.10 | 50.25±22.33|
|                              |            |        |        |     |     |      |       |      |    |
| **maze2d average**           | 18.67 | 30.65 | 149.30 | 16.35 | 91.66 | 176.96 | 151.48 | 162.27 | 60.74 |

#### Antmaze
| **Task-Name**|BC|BC-10%|TD3 + BC|CQL|IQL|AWAC|SAC-N|EDAC|DT |
|------------------------------|------------|--------|--------|-----|-----|------|-------|------|----|
|antmaze-umaze-v0 | 71.25±9.07 | 79.50±2.38 | 97.75±1.50 | 76.75±4.99 | 87.00±2.94 | 74.75±8.77 | 0.00±0.00 | 75.00±27.51 | 60.50±3.11|
|antmaze-medium-play-v0 | 4.75±2.22 | 8.50±3.51 | 6.00±2.00 | 1.75±0.96 | 86.00±2.16 | 14.00±11.80 | 0.00±0.00 | 0.00±0.00 | 0.25±0.50|
|antmaze-large-play-v0 | 0.75±0.50 | 11.75±2.22 | 0.50±0.58 | 0.00±0.00 | 53.00±6.83 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00|
|                              |            |        |        |     |     |      |       |      |    |
| **antmaze average**           | 25.58 | 33.25 | 34.75 | 26.17 | 75.33 | 29.58 | 0.00 | 25.00 | 20.25 |

## Citing CORL

If you use CORL in your work, please use the following bibtex
```bibtex
@article{tarasov2022corl,
  title={CORL: Research-oriented Deep Offline Reinforcement Learning Library},
  author={Tarasov, Denis and Nikulin, Alexander and Akimov, Dmitry and Kurenkov, Vladislav and Kolesnikov, Sergey},
  journal={arXiv preprint arXiv:2210.07105},
  year={2022}
}
```
