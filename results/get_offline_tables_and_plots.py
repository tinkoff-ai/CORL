import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rliable import library as rly, metrics, plot_utils

dataframe = pd.read_csv("runs_tables/offline_urls.csv")
with open("bin/offline_scores.pickle", "rb") as handle:
    full_scores = pickle.load(handle)

os.makedirs("./out", exist_ok=True)


def get_average_scores(scores):
    avg_scores = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    stds = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    for algo in scores:
        for data in scores[algo]:
            sc = scores[algo][data]
            if len(sc) > 0:
                ml = min(map(len, sc))
                sc = [s[:ml] for s in sc]
                scores[algo][data] = sc
                avg_scores[algo][data] = np.mean(sc, axis=0)
                stds[algo][data] = np.std(sc, axis=0)

    return avg_scores, stds


def get_max_scores(scores):
    avg_scores = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    stds = {algo: {ds: None for ds in scores[algo]} for algo in scores}
    for algo in scores:
        for data in scores[algo]:
            sc = scores[algo][data]
            if len(sc) > 0:
                ml = min(map(len, sc))
                sc = [s[:ml] for s in sc]
                scores[algo][data] = sc
                max_scores = np.max(sc, axis=1)
                avg_scores[algo][data] = np.mean(max_scores)
                stds[algo][data] = np.std(max_scores)

    return avg_scores, stds


def get_last_scores(avg_scores, avg_stds):
    last_scores = {
        algo: {
            ds: avg_scores[algo][ds][-1] if avg_scores[algo][ds] is not None else None
            for ds in avg_scores[algo]
        }
        for algo in avg_scores
    }
    stds = {
        algo: {
            ds: avg_stds[algo][ds][-1] if avg_stds[algo][ds] is not None else None
            for ds in avg_scores[algo]
        }
        for algo in avg_scores
    }
    return last_scores, stds


avg_scores, avg_stds = get_average_scores(full_scores)
max_scores, max_stds = get_max_scores(full_scores)
last_scores, last_stds = get_last_scores(avg_scores, avg_stds)


def add_domains_avg(scores):
    for algo in scores:
        locomotion = [
            scores[algo][data]
            for data in [
                "halfcheetah-medium-v2",
                "halfcheetah-medium-replay-v2",
                "halfcheetah-medium-expert-v2",
                "hopper-medium-v2",
                "hopper-medium-replay-v2",
                "hopper-medium-expert-v2",
                "walker2d-medium-v2",
                "walker2d-medium-replay-v2",
                "walker2d-medium-expert-v2",
            ]
        ]
        antmaze = [
            scores[algo][data]
            for data in [
                "antmaze-umaze-v2",
                "antmaze-umaze-diverse-v2",
                "antmaze-medium-play-v2",
                "antmaze-medium-diverse-v2",
                "antmaze-large-play-v2",
                "antmaze-large-diverse-v2",
            ]
        ]
        maze2d = [
            scores[algo][data]
            for data in [
                "maze2d-umaze-v1",
                "maze2d-medium-v1",
                "maze2d-large-v1",
            ]
        ]

        adroit = [
            scores[algo][data]
            for data in [
                "pen-human-v1",
                "pen-cloned-v1",
                "pen-expert-v1",
                "door-human-v1",
                "door-cloned-v1",
                "door-expert-v1",
                "hammer-human-v1",
                "hammer-cloned-v1",
                "hammer-expert-v1",
                "relocate-human-v1",
                "relocate-cloned-v1",
                "relocate-expert-v1",
            ]
        ]

        scores[algo]["locomotion avg"] = np.mean(locomotion)
        scores[algo]["antmaze avg"] = np.mean(antmaze)
        scores[algo]["maze2d avg"] = np.mean(maze2d)
        scores[algo]["adroit avg"] = np.mean(adroit)

        scores[algo]["total avg"] = np.mean(
            np.hstack((locomotion, antmaze, maze2d, adroit))
        )


add_domains_avg(last_scores)
add_domains_avg(max_scores)

algorithms = [
    "BC",
    "10% BC",
    "TD3+BC",
    "AWAC",
    "CQL",
    "IQL",
    "ReBRAC",
    "SAC-N",
    "EDAC",
    "DT",
]
datasets = dataframe["dataset"].unique()
ordered_datasets = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
    "locomotion avg",
    "maze2d-umaze-v1",
    "maze2d-medium-v1",
    "maze2d-large-v1",
    "maze2d avg",
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-large-play-v2",
    "antmaze-large-diverse-v2",
    "antmaze avg",
    "pen-human-v1",
    "pen-cloned-v1",
    "pen-expert-v1",
    "door-human-v1",
    "door-cloned-v1",
    "door-expert-v1",
    "hammer-human-v1",
    "hammer-cloned-v1",
    "hammer-expert-v1",
    "relocate-human-v1",
    "relocate-cloned-v1",
    "relocate-expert-v1",
    "adroit avg",
    "total avg",
]

"""# Tables"""


def get_table(
    scores,
    stds,
    pm="$\\pm$",
    delim=" & ",
    row_delim="\\midrule",
    row_end=" \\\\",
    row_begin="",
):
    rows = [row_begin + delim.join(["Task Name"] + algorithms) + row_end]
    prev_env = "halfcheetah"
    for data in ordered_datasets:
        env = data.split("-")[0]
        if env != prev_env:
            if len(row_delim) > 0:
                rows.append(row_delim)
            prev_env = env

        row = [data]

        for algo in algorithms:
            if data in stds[algo]:
                row.append(f"{scores[algo][data]:.2f} {pm} {stds[algo][data]:.2f}")
            else:
                row.append(f"{scores[algo][data]:.2f}")
        rows.append(row_begin + delim.join(row) + row_end)
    return "\n".join(rows)


print(get_table(last_scores, last_stds))
print()
print(get_table(max_scores, max_stds))
print()
print(get_table(last_scores, last_stds, "±", "|", "", "|", "|"))
print()
print(get_table(max_scores, max_stds, "±", "|", "", "|", "|"))

os.makedirs("out", exist_ok=True)

plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["figure.dpi"] = 300
sns.set(style="ticks", font_scale=1.5)


linestyles = [
    ("solid", "solid"),
    ("dotted", (0, (1, 1))),
    ("long dash with offset", (5, (10, 3))),
    ("densely dashed", (0, (5, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]

for data in datasets:
    min_score = 1e6
    max_score = -1e6
    for i, algo in enumerate(algorithms):
        if avg_scores[algo][data] is not None:
            to_draw = avg_scores[algo][data]
            std_draw = avg_stds[algo][data]
            if len(to_draw) == 600 or len(to_draw) == 601:
                to_draw = to_draw[::3]
                std_draw = std_draw[::3]
            if len(to_draw) == 1000:
                to_draw = to_draw[::5]
                std_draw = std_draw[::5]
            if len(to_draw) == 3000:
                to_draw = to_draw[::15]
                std_draw = std_draw[::15]
            steps = np.linspace(0, 1, len(to_draw))
            min_score = min(min_score, np.min(to_draw))
            max_score = max(max_score, np.max(to_draw))
            plt.plot(
                steps, to_draw, label=algo, linestyle=linestyles[i % len(linestyles)][1]
            )
            plt.fill_between(steps, to_draw - std_draw, to_draw + std_draw, alpha=0.1)

    plt.title(data)
    plt.xlabel("Fraction of total steps")
    plt.ylabel("Normalized score")
    plt.ylim([min_score - 3, max_score + 3])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.savefig(f"out/{data}.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


def convert_dataset_name(name):
    name = name.replace("v2", "")
    name = name.replace("v1", "")
    name = name.replace("v0", "")
    name = name.replace("medium-", "m-")
    name = name.replace("umaze-", "u-")
    name = name.replace("large-", "l-")
    name = name.replace("replay-", "re-")
    name = name.replace("random-", "ra-")
    name = name.replace("expert-", "e-")
    name = name.replace("play-", "p-")
    name = name.replace("diverse-", "d-")
    name = name.replace("human-", "h-")
    name = name.replace("cloned-", "c-")
    return name[:-1]


def plot_bars(scores, save_name):
    agg_l = []

    for env in [
        "halfcheetah",
        "hopper",
        "walker2d",
        "maze2d",
        "antmaze",
        "pen",
        "door",
        "hammer",
        "relocate",
    ]:
        if env in ["halfcheetah", "hopper", "walker2d"]:
            datas = ["medium-v2", "medium-expert-v2", "medium-replay-v2"]
        elif "maze2d" in env:
            datas = ["umaze-v1", "medium-v1", "large-v1"]
        elif "antmaze" in env:
            datas = [
                "umaze-v2",
                "umaze-diverse-v2",
                "medium-play-v2",
                "medium-diverse-v2",
                "large-play-v2",
                "large-diverse-v2",
            ]
        else:
            datas = ["human-v1", "cloned-v1", "expert-v1"]
        for data in datas:
            line = convert_dataset_name(f"{env}-{data}")
            for algo in algorithms:
                agg_l.append([algo, line, scores[algo][f"{env}-{data}"]])
    df_agg = pd.DataFrame(agg_l, columns=["Algorithm", "Dataset", "Normalized Score"])

    sns.set(style="ticks", font_scale=2)
    plt.rcParams["figure.figsize"] = (20, 10)  # (10, 6)

    b = sns.barplot(
        data=df_agg[
            df_agg.Dataset.apply(
                lambda x: "cheetah" in x or "hopper" in x or "walker" in x
            )
        ],
        x="Dataset",
        y="Normalized Score",
        hue="Algorithm",
    )
    plt.grid()
    # plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"out/bars_{save_name}_loco.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    b = sns.barplot(
        data=df_agg[df_agg.Dataset.apply(lambda x: "maze2d" in x)],
        x="Dataset",
        y="Normalized Score",
        hue="Algorithm",
    )
    # plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    plt.grid()

    plt.savefig(f"out/bars_{save_name}_maze.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    b = sns.barplot(
        data=df_agg[df_agg.Dataset.apply(lambda x: "ant" in x)],
        x="Dataset",
        y="Normalized Score",
        hue="Algorithm",
    )
    # plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    plt.grid()

    plt.savefig(f"out/bars_{save_name}_ant.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

    b = sns.barplot(
        data=df_agg[
            df_agg.Dataset.apply(
                lambda x: "pen" in x or "hammer" in x or "door" in x or "relocate" in x
            )
        ],
        x="Dataset",
        y="Normalized Score",
        hue="Algorithm",
    )
    plt.grid()
    # plt.tight_layout()
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    sns.move_legend(b, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(f"out/bars_{save_name}_adroit.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()


plot_bars(last_scores, "last")

plot_bars(last_scores, "max")


def flatten(data):
    res = {}
    for algo in data:
        flat = []
        for env in data[algo]:
            if "avg" not in env:
                env_list = np.array(data[algo][env])[:, -1]
                flat.append(env_list)
        res[algo] = np.array(flat).T
    return res


flat = flatten(full_scores)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 300
sns.set(style="ticks", font_scale=0.5)
plt.rcParams.update(
    {
        # "font.family": "serif",
        "font.serif": "Times New Roman"
    }
)
# sns.set_palette("tab19")

algorithms = list(flat)

normalized_score_dict = flat

# Human normalized score thresholds
thresholds = np.linspace(-5.0, 150.0, 31)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    normalized_score_dict, thresholds
)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
# plt.legend()
plot_utils.plot_performance_profiles(
    score_distributions,
    thresholds,
    performance_profile_cis=score_distributions_cis,
    colors=dict(zip(algorithms, sns.color_palette("colorblind"))),
    xlabel=r"D4RL Normalized Score $(\tau)$",
    ax=ax,
    legend=True,
)
plt.savefig("out/perf_profiles_offline.pdf", dpi=300, bbox_inches="tight")
plt.close()

algorithm_pairs = {}
sns.set(style="ticks", font_scale=0.5)
algs = ["IQL", "AWAC", "EDAC", "SAC-N", "CQL", "TD3+BC", "DT", "BC", "10% BC"]
for a1 in ["ReBRAC"]:
    for a2 in algs:
        algorithm_pairs[f"{a1},{a2}"] = (flat[a1], flat[a2])
average_probabilities, average_prob_cis = rly.get_interval_estimates(
    algorithm_pairs, metrics.probability_of_improvement, reps=200
)
ax = plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis)
# ax.set_xlim(0.5, 0.8)
plt.savefig("out/improvement_probability_offline.pdf", dpi=300, bbox_inches="tight")
plt.close()
