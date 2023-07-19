# CORL Contribution Guidelines

We welcome:

- Bug reports
- Pull requests for bug fixes
- Logs and documentation improvements
- New algorithms and datasets
- Better hyperparameters (but with proofs)

## Contributing to the codebase

Contributing code is done through standard github methods:

```commandline
git clone git@github.com:tinkoff-ai/CORL.git
cd CORL
pip install -r requirements/requirements_dev.txt
```

1. Fork this repo
2. Make a change and commit your code
3. Submit a pull request. It will be reviewed by maintainers and they'll give feedback or make requests as applicable

### Code style

The CI will run several checks on the new code pushed to the CORL repository. 
These checks can also be run locally without waiting for the CI by following the steps below:
1. [install `pre-commit`](https://pre-commit.com/#install),
2. install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit. The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`. **Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.

We use [Ruff](https://github.com/astral-sh/ruff) as our main linter. If you want to see possible problems before pre-commit, you can run `ruff check --diff .` to see exact linter suggestions and future fixes.

## Adding new algorithms

All new algorithms should go to the `algorithms/contrib`. 
We as a team try to keep the core as reliable and reproducible as possible, 
but we may not have the resources to support all future algorithms. 
Therefore, this separation is necessary, as we cannot guarantee that all 
algorithms from `algorithms/contrib` exactly reproduce the results of their original publications.

Make sure your new code is properly documented and all references to the original implementations and papers are present (for example as in [Decision Transformer](algorithms/offline/dt.py)). 
Please, *explain all the tricks and possible differences from the original implementation in as much detail as possible*. 
Keep in mind that this code may be used by other researchers. Make their lives easier!

### Considerations
While we welcome any algorithms, it is better to open an issue with the proposal before 
so we can discuss the details. Unfortunately, not all algorithms are equally 
easy to understand and reproduce. We may be able to give a couple of advices to you,
or on the contrary warn you that this particular algorithm will require too much 
computational resources to fully reproduce the results and it is better to do something else.

### Running benchmarks

Although you will have to do a hyperparameter search while reproducing the algorithm, 
in the end we expect to see final configs in `configs/contrib/<algo_name>/<dataset_name>.yaml` with the best hyperparameters for all calculated 
datasets. The configs should be in yaml format, containing all parameters sorted 
in alphabetical order (see existing configs for an inspiration).

Use this conventions to name your runs in the configs:
1. `name: <algo_name>`
2. `group: <algo_name>-<dataset_name>-multiseed-v0`. Increment version if needed
3. use our [\_\_post_init\_\_](https://github.com/tinkoff-ai/CORL/blob/962688b405f579a1ce6ec1b57e6369aaf76f9e69/algorithms/offline/awac.py#L48) implementation in your config dataclass

Since we are releasing wandb logs for all algorithms, you will need to submit multiseed (4 seeds) 
training runs the `CORL` project in the wandb [corl-team](https://wandb.ai/corl-team) organization. We'll invite you there when the time will come.

We usually use wandb sweeps for this. You can use this example config (it will work with pyrallis as it expects `config_path` cli argument):
```yaml
# sweep_config.yaml
entity: corl-team
project: CORL
program: algorithms/contrib/<algo_name>.py
method: grid
parameters:
  config_path:
    values: [
        "configs/contrib/<algo_name>/<dataset_name_1>.yaml",
        "configs/contrib/<algo_name>/<dataset_name_2>.yaml",
        "configs/contrib/<algo_name>/<dataset_name_3>.yaml",
    ]
  train_seed:
    values: [0, 1, 2, 3]
```

Then proceed as usual. Create wandb sweep with `wandb sweep sweep_config.yaml`, then run agents with `wandb agent <agent_id>`.

### Checklist

- [ ] Issue about new algorithm is open
- [ ] Single-file implementation is added to the `algorithms/contrib`
- [ ] PR has passed all the tests
- [ ] Evidence that implementation reproduces original results is provided
- [ ] Configs with best hyperparameters for all datasets are added to the `configs/contrib`
- [ ] Logs for best hyperparameters are submitted to the our wandb organization
