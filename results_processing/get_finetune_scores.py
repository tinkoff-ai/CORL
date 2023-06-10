import os
import pickle

import pandas as pd
from tqdm import tqdm
import wandb

dataframe = pd.read_csv("runs_tables/finetune_urls.csv")

api = wandb.Api(timeout=29)


def get_run_scores(run_id, is_dt=False):
    run = api.run(run_id)
    score_key = None
    full_scores = []
    regret = None
    max_dt = -1e10

    for k in run.history().keys():
        if "normalized" in k and "score" in k and "std" not in k:
            if is_dt:
                st = k
                if "eval/" in st:
                    st = st.replace("eval/", "")
                target = float(st.split("_")[0])
                if target > max_dt:
                    max_dt = target
                    score_key = k
            else:
                score_key = k
                break
    for i, row in run.history(keys=[score_key], samples=5000).iterrows():
        full_scores.append(row[score_key])
    for i, row in run.history(keys=["eval/regret"], samples=5000).iterrows():
        if "eval/regret" in row:
            regret = row["eval/regret"]
    l = len(full_scores) // 2
    return full_scores[:l], full_scores[l:], regret


def process_runs(df):
    algorithms = df["algorithm"].unique()
    datasets = df["dataset"].unique()
    full_scores = {algo: {ds: [] for ds in datasets} for algo in algorithms}
    for index, row in tqdm(
            df.iterrows(), desc="Runs scores downloading", position=0, leave=True
    ):
        full_scores[row["algorithm"]][row["dataset"]].append(
            get_run_scores(row["url"], row["algorithm"] == "DT")
        )
    return full_scores


full_scores = process_runs(dataframe)

os.makedirs("bin", exist_ok=True)
with open("bin/finetune_scores.pickle", "wb") as handle:
    pickle.dump(full_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
