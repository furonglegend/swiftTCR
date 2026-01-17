import itertools
import subprocess


def run_ablation(config_template, grid):
    """
    Automatic ablation over hyperparameter grid.
    """
    keys = grid.keys()
    for values in itertools.product(*grid.values()):
        cfg = dict(zip(keys, values))
        print("Running ablation:", cfg)
        subprocess.call(["python", "run_main.py", "--config", config_template])
