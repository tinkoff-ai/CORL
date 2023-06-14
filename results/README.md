# Reproducing figures and tables

To reproduce all figures and tables from the paper, do the following steps:
1. Run `get_{offline, finetune}_urls.py` if needed. These scripts collect all wandb logs into .csv files and save them into the `runs_tables` folder. We provide the tables, but you can recollect them.
2. Run `get_{offline, finetune}_scores.py` if needed. These scripts collect data from runs kept in .csv files and save evaluation scores (and regret in case of offline-to-online) into pickled files, which are stored in the `bin` folder. We provide the pickled data, but if you need to extract more data, you can modify scripts for your purposes.
3. Run `get_{offline, finetune}_tables_and_plots.py`. These scripts use pickled data, print all the tables, and save all figures into the `out` directory.
