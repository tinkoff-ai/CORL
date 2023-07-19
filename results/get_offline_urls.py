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
        elif "dataset_name" in run.config:
            dataset = run.config["dataset_name"]
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


get_urls("tlab/CORL/sweeps/vs7dn9cw", "BC")

get_urls("tlab/CORL/sweeps/n0cbyj25", "BC")

get_urls("tlab/CORL/sweeps/uooh3e8g", "AWAC")

get_urls("tlab/CORL/sweeps/ttrkt97z", "TD3+BC")

get_urls("tlab/CORL/sweeps/tdn3t7wv", "IQL")

get_urls("tlab/CORL/sweeps/0ey8ru2j", "DT")

get_urls("tlab/CORL/sweeps/fdxa3fga", "SAC-N")

get_urls("tlab/CORL/sweeps/ptgj7mhu", "EDAC")

get_urls("tlab/CORL/sweeps/9828rg17", "BC")

get_urls("tlab/CORL/sweeps/b49ukhml", "AWAC")

get_urls("tlab/CORL/sweeps/pprq5ur4", "SAC-N")

get_urls("tlab/CORL/sweeps/je3ac4nx", "EDAC")

get_urls("tlab/CORL/sweeps/ymvugomt", "IQL")

get_urls("tlab/CORL/sweeps/qkukub0n", "TD3+BC")

get_urls("tlab/CORL/sweeps/nvn328sg", "DT")

get_urls("tlab/CORL/sweeps/2d0jczwg", "IQL")

get_urls("tlab/CORL/sweeps/9jtj053n", "TD3+BC")

get_urls("tlab/CORL/sweeps/zzglpp0f", "BC")

get_urls("tlab/CORL/sweeps/hp7tiw93", "CQL")

get_urls("tlab/CORL/sweeps/3ui4jhet", "SAC-N")

get_urls("tlab/CORL/sweeps/uhgujgoy", "AWAC")

get_urls("tlab/CORL/sweeps/0v8tnh8y", "BC")

get_urls("tlab/CORL/sweeps/e1e6fzv1", "BC")

get_urls("tlab/CORL/sweeps/sg1hx5v7", "DT")

get_urls("tlab/CORL/sweeps/nev3j9wx", "EDAC")

get_urls("tlab/CORL/sweeps/03sfog1g", "ReBRAC")

# OLD RUNS
# BC
collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/gae6mjr6")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3dda9gfw")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3sgbj9n0")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/67eno4ma")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3bur5hke")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/330z0l2v")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/1i05t3vj")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/k9yfle3x")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/1zreo8zw")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/18vbgvb2")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/ky3vncuf")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/3tz0z6nh")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/31dmbfoz")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1rhop7f6")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2q070txr")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/sbcrq218")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/28iujcoa")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/2f12hcq3")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1ptuak40")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/36y8187b")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3bn0h2zy")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3joz13bc")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3s9l1a83")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1q966noh")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2b85pbgd")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/ca0nxbh4")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1ipey1bk")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/x35k6x12")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1owdjob7")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/xoosoz9n")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3r09yx27")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3k5v2mso")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/39tqleqs")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/9cddvu7a")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/17v5isiw")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2a8wzq2t")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1tgqpiks")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/19yfj5xu")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2bneh6uw")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3twop214")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/rhkaisgq")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/287bzpdd")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/l2gfzbhg")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3gnugxzy")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2uwtj2md")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/60yn1nfx")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2p0w55iq")

collected_urls["algorithm"].append("BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2rv6pvln")

# 10% BC
collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/84b74c6e-bc52-4083-a601-6a387726c61d")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/e22c302b-e387-4d12-a498-db1c7b787306")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/c76a5b7c-f459-498e-9aa9-6c0366ded313")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/dafaa4dc-9359-4feb-be9b-39c3dcadcdd4")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/7aff87ac-17e1-49a8-b52d-a210c9be9eee")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/d14de446-beea-413f-ad5e-c90dfd0e790c")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/d4713f18-520a-459e-80a6-0acd70d0710f")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/dfbcb740-26ca-4bbf-9065-ad3ecd60c261")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/134273d4-5eb7-4e42-a62b-b3a387a7a2a4")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/f6b33b84-b8c4-42a9-aae4-0d12db4f8b92")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/c8dff5d6-4b22-4e7f-a3b1-5913ae9b0aed")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/6d454981-bf52-4126-b4bc-436e566b76be")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/5d7df542-1567-462f-8885-8c8a0e8a5d19")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/d1d0f883-1b1d-4429-8c3c-02de6c989cdb")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/8ccf19da-a0e6-4267-a53a-276349aea3be")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2c0ea1a2-614b-414a-b6fc-baa9663891da")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3cc3a7f7-8ff0-497c-a6e0-e6c5c5ca9688")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c0de3f56-a236-44a4-a532-04064af81b18")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/f2f1507a-9066-4df1-962e-a3d9bed3015a")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/6313d5cf-9158-4585-9f48-cccbe1ff16f1")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/ba6e7a6d-2548-4d8a-a35f-286782c3658e")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/ab521663-97d4-4b00-a992-b602d495f7d7")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/f6c1e15a-23d4-472d-846f-e766a835d67b")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/482908a6-eb2e-4b3d-8254-0ef0124f488e")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/7fc8e114-0c73-4c47-977a-7f8d337dac1f")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/c2e4d867-a355-4030-b23f-e9845da0c4bf")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/cec9a1e2-a270-4270-861b-88535dcd4103")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/dcc5696c-bc69-41a3-a4f7-2865a16651ef")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/b86f27c4-05d0-43d8-b95e-b81edeb45144")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/364433ae-2974-48c7-a8e5-fc7606dbc819")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/ba1ae355-2945-4c82-a7be-49e421b59574")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c9b94c6c-8a73-4259-848b-61f7b9386309")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/323e3a40-e919-4dd6-9d97-3e6f7a01b118")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/6065ffc6-8cee-45d8-b2e5-a600922a89cc")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/b418e6f1-1fcc-43dc-b5e3-475c17d3da1a")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/9b7add9a-d916-4ac8-9538-09d82ea6a7c4")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/0155fffe-76ae-4580-ba4a-c90d8c83c8d6")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/e7ea6fec-ac94-483f-af5a-c20790569efd")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/af373d51-823c-4ebc-b863-3ffefb6ad5f0")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/82e587c5-afc5-47f3-b71c-734472174a19")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1bca103d-fa9b-405f-a4c3-f4f5aee161c1")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/706ea73c-c148-4f2f-96c6-347e600ae566")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/e51f8235-0ea3-4eb5-a2ff-67d159404783")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/5cd02078-1a5b-4721-9070-c8a5d7bce477")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/40eaf786-7305-46a0-8b4c-2dc608c9cf34")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/4bceaa03-d8e6-4ec5-b417-d1007f4a7504")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/e1f340a7-f659-4143-8c76-22d341532e9c")

collected_urls["algorithm"].append("10% BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/df22f73b-3904-4d3d-be82-8565a94f90a9")

#
collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3gmwuspv")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/hfnz06jo")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/22zd4qy5")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/2je1ydbq")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/2cn5kybz")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/4wfevsn1")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/8uc5g9vl")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3q3i7kr4")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/1383sspe")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/ujqk6bcx")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/2har775v")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/1t9zpxwq")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1manw8ou")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/glmwyvtm")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/99lixj21")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/21qd6jdk")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/13i7gvdv")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/lfnzn3ek")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/2iqxrf7v")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/28q8k0is")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2klwm3m9")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/vgj8gxc9")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1zpikd1i")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3mhuu91m")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/o9cy1xot")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/9oorg18b")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/8umnr31k")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/8ay8wua0")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/36r6bciu")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3dhx3yws")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/2xgt4p29")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/2i8f6fsw")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1pocua7w")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3apac4jp")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3axkszn9")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/iyy3p627")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2evz37in")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/rcuf9ji6")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2nguxmuw")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/563x3nqx")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3pp38z95")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c7htx54f")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/35i1e9k3")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/34kpercv")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1y6a1ghl")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1r5ja7w3")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2ksjowc8")

collected_urls["algorithm"].append("TD3+BC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1v789w9r")

#
collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/f5447eae-38f5-404e-ab97-979d12a62dba")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/063ec049-6092-46fd-8d06-5c43aa0c8933")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/517996bc-48dd-4cc5-a1a2-b599668dfb03")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/cdb110c8-baed-4b72-9338-e2df069c1999")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/863ba3ad-2e15-4027-a561-50a1ce837a2e")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/a120a194-2a4d-493f-a105-29e81c2167f3")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/db99a51a-20ec-4898-b432-7bed581b11eb")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/ef619bf1-e43f-4ca0-b26a-e44a79c8d6c4")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/d61f15f2-bb63-4b0e-8a3f-0a8397f85c99")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/bc356f6c-ff8a-4fcb-8f7d-eda711bf187f")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/e55c1f59-4a22-4adf-90db-55b761184c31")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/754eb9df-300b-4816-b483-1ecc8630d170")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/fcdf10b7-3f06-4950-89e5-0bb706d32fa2")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3149b249-61b7-42b5-b62c-560263073ceb")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/e3f4068c-2f7a-4d98-8bfe-71e5bcd37f60")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/07bafbb5-cef0-487f-9d18-43f5e6f41e5b")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/bdc16cb0-7ba1-44e5-a634-f7821849e911")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1c63037a-0f9e-4c92-8e30-f868e5899235")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/49ccdf3d-49f8-43f7-ae5e-5f2166928b08")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/86e2bdf2-bfc8-4dd8-b245-06f3c5948525")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/b7865c5a-6382-4dfe-967d-f5f41caef859")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1a9ae20a-0ef3-4517-aa21-0114606e8e44")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/68993e5b-f477-496e-ab8c-da7808851e31")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/d9682650-69b2-4cce-832c-a0a5d63d7b87")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/51b5a164-e6ab-4929-bf76-b786a3e40654")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/abd10b19-e2c5-4e27-99ac-2ca8445acd51")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/5c0c2cb0-2457-40dc-905b-8bf32b8a75fe")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/98977940-fab9-462c-ac70-3fcd10bc55cb")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/a513ea52-a879-47a6-ab4c-ac1a046b5cc2")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/0cffd41b-d983-4b45-93c8-2e22fc5801c0")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c7b8a1c8-170f-4060-860c-62553ff67911")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/7df0497b-d805-47ce-91ba-485d7bff6fb6")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3db49470-beba-49f8-963b-bc7fbe79d107")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/21fea44e-168d-4356-a72c-1ac09a482d05")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/60a8e98b-5933-491e-83c7-f48b777fb52e")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/7eaf035d-9394-4eee-97f0-50347b108b6a")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/76b97aeb-4327-4fb1-bbd4-572f84b9ac6c")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2eaf20df-c7d2-42c7-9d6f-5f29e240b99f")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/fa033830-cec7-4144-894d-741391fdb81d")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/04917eeb-b7a5-4e02-9e89-7eed774cd00b")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/d296d6ef-8a37-4c39-be14-ab54eb85a0ee")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/825a83d5-0ed4-4c97-9c79-13edfa43e6cc")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/277df654-7035-4469-8150-ff3df3f6230e")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/6428588e-c9bc-43ba-a945-285248e0664b")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/0d1ae046-abcb-4da1-b2d3-1360bbd8f54f")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/9eb231d9-6c25-4d42-9564-90164b7e680b")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/f4c212ba-7b8e-428e-9953-71606fd84d67")

collected_urls["algorithm"].append("DT")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3bc164b8-1fc0-4ce5-a32d-701e522ad5b1")

#
collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/a7e3d2a0-2dbc-4eba-b28d-8315f992bae3")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/65981364-10fc-47d3-bb35-ccc67254ca23")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/ceb4bd07-50d4-426c-9e2b-a54fc4a1092a")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/a2fe5d76-b680-42b1-aafa-4f7fae8e9575")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/342b9c5e-eb78-45b1-99fc-97654d2d619a")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/eaab4d73-b002-4587-89e9-b101efc5c385")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/f83b4b8c-bddd-469a-acf5-c2c59b80fd3c")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/4c2065f4-e773-4760-a045-18958aff4685")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/eef336bc-42f0-46bc-90df-17d6b5647263")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/16b37de3-9011-4a20-b58a-d1d97946125a")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/81bdccf5-1ce7-4ab5-9228-1193209b9f85")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/700bc2bd-3ae8-4845-a5a7-ea9ce5a5bf68")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/c0015d64-2bce-4bf7-a804-92390d022ec9")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/f7a045fb-89de-4df1-a827-0b0aff6fa803")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/c61cc412-51fa-41ef-be06-5e8eaba5272e")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/e08593b0-edc7-49a7-bf68-e66e613ed20f")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3be8a859-82e5-4cc2-899d-4ff7f88a90ed")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c5dd3800-eed4-4711-8172-0d22bc985ed9")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/ff761882-9f47-4f3b-8cf9-0f5cf0b40339")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/0257eae7-716d-4c68-b8a2-1d99c74d79d0")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/8c18b80d-028d-48dd-a371-b2fab308469a")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/c86ba1cc-8b4c-4dd8-b64d-8f57a8131d95")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/bc5fda0c-2f5c-4391-8bd5-c4f2e15c2e0c")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/c3fdffef-f3cb-4d18-9d94-af4e0651ba21")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/95c7d8e0-f634-403a-8edb-ea00afd5c69c")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/4580d97f-15b0-4d54-887c-91cf0a3368ea")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/ad47291b-1469-48b5-ba20-266a05bc9326")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/16f77985-8033-4953-8066-c33c49141581")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/69bf1797-94b0-43fa-b22c-a6406a93d222")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/dadbb413-ae11-48bb-a4bb-94c8b4c7d53f")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c1db8aa9-9bfc-4687-a8b5-6096c90f6e9b")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/b6ff762e-c0be-4b6d-ac23-8b5ffcb28a56")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/ab688db2-ab1d-4d96-ba40-6186c7ecb16b")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/d0a5c6be-7b64-4ddb-b965-1ae8e0533363")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/9f67f421-c55b-4527-8ea0-8e6579a3bb61")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/ab44a4d1-6aee-420e-b691-307bd083d2ea")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3394eb73-a8b3-463c-9a57-8dd65833ecdd")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/89527361-8f90-47a5-8882-ac3459de0d0a")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/f02528e5-86d6-4242-961f-106cb0e5df14")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/132a99bc-386a-4eb4-a64c-74699d0563b5")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/33ce900d-b858-4bc3-a6dc-71f9615cfad5")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/87addd3a-42bd-45b7-8dcb-a921dfa6dad5")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/bcfb639c-1d44-4228-bbd8-e560b48bb5d6")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/249f88e4-c98f-401f-bb36-4d5f239fff74")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/fc7fa907-ab00-457d-a00d-2bdd65688379")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/20f7258d-0f07-4002-86b2-4c3ec65ee067")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/c3e71147-80a2-4ae8-bb59-9b994daaa516")

collected_urls["algorithm"].append("SAC-N")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/e36a72da-482f-4a70-803f-1a0d7eccb265")

#
collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/1m3k2bd1")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3jzf46zg")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/exlzrv4v")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3r2qku3k")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3crj1urn")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/25vxky59")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/258aw9fy")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3oc7jc1q")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/31ak0z9b")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/hjl7pxfa")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/2qq9dfgc")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/c0pdrw6f")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/5d588f87-fe51-4253-b310-a75fbf8d3702")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/10aa52ac-b2f4-43c4-97f1-4bee57fdab24")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3500687d-84c6-4cc6-88a9-ac432fe83f42")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2108ebe3-d55d-418a-9fda-f78a8337909a")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/8853c87c-9bdc-411e-8128-f0976c510485")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/b86adeb5-282b-4f9b-bd4f-361b576c9988")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/6b675ca0-3fed-498a-ae54-e964673158d4")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/48813224-53a2-495e-86a2-d72a5b95ba94")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/996be0e1-ae88-492d-b261-15f034cc6203")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/62bcf801-db79-438e-b0f4-74436f3c67b1")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/ffddfea8-2e9b-493b-88df-04a15f97d7a8")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/b07eb900-8653-4688-a10f-111f3eb3c84a")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/59f743f9-3b3a-4306-83b5-98721508bf2f")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/74a7e942-ca43-44e8-85f7-976fa7dd2edd")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/20425c80-a0f3-4e1a-9991-a85db7012417")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/6fb1e9e2-9485-40c9-ac77-b118cd9cc55b")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/6145c71a-ce9b-4817-bf94-a6eef9b79377")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c7d59200-7e0f-47a4-846a-123fb23d3c30")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/00379327-06d9-4117-9abb-0f4fef0d6f38")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/dc1c3646-d8fd-4671-b43c-b987441f70cf")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/a58fedea-d5fe-4481-bca4-0e44989f049e")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/05dc4e17-4c73-4f71-b5c3-2eb39aae36c8")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/155aa581-5e1f-4d32-acd5-edde7c5e3c6a")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/5e5e6d1a-59c4-4044-9d50-7d1b920bb626")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/ffb22753-338f-4d2a-ba45-aaeba6a5eed3")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/6d1e8c3f-bd50-4e02-8adc-bf7db13d15ad")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/f99181eb-499d-48be-b1e3-5349f8fe3731")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/fd8b7f41-48cc-4578-8fc8-55ec5e5884df")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/a0a92721-04b1-4868-809e-2ce37358516b")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/c484e9cd-ee4d-427a-941d-80926caa3128")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/5790cb46-ea8c-42b6-abe6-a70faa0f4633")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/ed665d8c-1bb5-4858-9136-574bf523b39a")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1e6e9a77-a335-41e0-9e29-6271f5a4fcda")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/d6492463-82f1-4512-99fa-b23073d6b418")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/96027203-781b-46ee-bf59-e565227f2f7b")

collected_urls["algorithm"].append("EDAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/d5f5f415-9d1b-4d35-b4e5-c1cf278af46c")

#
collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3me14n0w")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/8671xq2j")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3keq4k8a")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-large-v1")
collected_urls["url"].append("tlab/CORL/runs/3jq85ti0")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/1vvutaak")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/16nzq1ng")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3552gil2")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-medium-v1")
collected_urls["url"].append("tlab/CORL/runs/3l3dpq11")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/3usi5cuh")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/2vvw9y8h")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/2vcog7cq")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("maze2d-umaze-v1")
collected_urls["url"].append("tlab/CORL/runs/qp93j6we")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1n8ttdck")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/1bpgemq2")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/39wb3kat")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/w9i9g39x")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3gfpaz8e")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3aerk47s")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/275nzj65")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/2fxchaks")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/220xo7sy")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/186848oq")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2qcui7s9")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("halfcheetah-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3izk7ats")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3p8nop3c")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2n4njt2r")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/cfgxmidd")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/o3jqikii")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1jg2th4m")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3qqk3v1v")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1og7e8w1")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/1hg2vtf9")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3b6t3c8p")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/i15nczq4")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3v7jt3p7")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("hopper-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2uvghydj")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3v1rznw2")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/2ov8rc9w")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3funjmu4")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-v2")
collected_urls["url"].append("tlab/CORL/runs/3o823qdi")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/21coamdv")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/35cmwtdl")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/3pvuqbr5")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-replay-v2")
collected_urls["url"].append("tlab/CORL/runs/ic2e00s6")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/2utgl834")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3hvawfk9")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/3mo9ld3q")

collected_urls["algorithm"].append("AWAC")
collected_urls["dataset"].append("walker2d-medium-expert-v2")
collected_urls["url"].append("tlab/CORL/runs/1aihv0tw")

dataframe = pd.DataFrame(collected_urls)
dataframe.to_csv("runs_tables/offline_urls.csv", index=False)
