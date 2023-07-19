import pandas as pd
import wandb

collected_urls = {
    "algorithm": [],
    "dataset": [],
    "url": [],
}


def get_urls(sweep_id, algo_name):
    s = sweep_id
    api = wandb.Api(timeout=39)
    sweep = api.sweep(s)
    runs = sweep.runs
    for run in runs:
        if "env" in run.config:
            dataset = run.config["env"]
        elif "env_name" in run.config:
            dataset = run.config["env_name"]
        name = algo_name
        if "10" in "-".join(run.name.split("-")[:-1]):
            name = "10% " + name
        if "medium" not in dataset:
            if "cheetah" in dataset or "hopper" in dataset or "walker" in dataset:
                continue
        if "v0" not in dataset and "dense" not in dataset:
            print(name, dataset, run.url)
            collected_urls["algorithm"].append(name)
            collected_urls["dataset"].append(dataset)
            collected_urls["url"].append(run.url.replace("https://wandb.ai/", ""))


get_urls("tlab/CORL/sweeps/7c42z4dz", "SPOT")

get_urls("tlab/CORL/sweeps/l3an1ck7", "AWAC")

get_urls("tlab/CORL/sweeps/snbq2jky", "CQL")

get_urls("tlab/CORL/sweeps/ucrmi909", "IQL")

get_urls("tlab/CORL/sweeps/efvz7d68", "Cal-QL")

dataframe = pd.DataFrame(collected_urls)

dataframe.to_csv("runs_tables/finetune_urls.csv", index=False)
