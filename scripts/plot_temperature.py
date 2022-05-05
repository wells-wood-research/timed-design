import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    results_arr = []
    column_names = ["model", "pdb", "seq", "charge", "isoelectric_point", "solubility", "expressivity", "temp"]
    # Loop through each model and temp:
    for model in args.models:
        for t in args.temperature:
            # Compose metrics and solubility files:
            metrics_file = args.input_path / f"{model}_temp_{t}_n_{args.sample_n}_metrics.csv"
            solubility_file = args.input_path / f"{model}_temp_{t}_n_{args.sample_n}.csv"
            # Check paths exist:
            assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist."
            assert solubility_file.exists(), f"Solubility file {solubility_file} does not exist."
            # Load files into an array:
            metrics_arr = np.genfromtxt(metrics_file, delimiter=",", dtype=str)
            solubility_arr = np.genfromtxt(solubility_file, delimiter=",", dtype=str, skip_header=1)
            # Check that shapes are compatible:
            assert metrics_arr.shape == solubility_arr.shape, f"Shapes of metrics and solubility have different shapes {metrics_arr.shape} and {solubility_arr.shape}."
            # Add model name and temperature
            model_arr = np.array([[model]*len(metrics_arr)]).T
            temp_arr = np.array([[t]*len(metrics_arr)]).T
            curr_metrics = np.concatenate((model_arr, metrics_arr, solubility_arr[:, 2:], temp_arr), axis=1)
            # Merge with other arrays:
            results_arr += curr_metrics.tolist()
    # Pool all results into one big array:
    results_arr = np.array(results_arr)
    # Load results in pandas
    df = pd.DataFrame(data=results_arr, columns=column_names)
    # Load as numbers: (genfromtxt workaround)
    df[["charge", "isoelectric_point", "solubility", "expressivity", "temp"]] = df[["charge", "isoelectric_point", "solubility", "expressivity", "temp"]].apply(pd.to_numeric)
    # Graph Solubility and Expressivity
    ax = sns.lineplot(x="temp", y="solubility", hue="model", data=df, palette="Set2", style="model", markers=True, dashes=False, ci='sd')
    ax.set(ylim=(0, 1))
    ax.savefig("solubility.png")
    plt.close()
    ax = sns.lineplot(x="temp", y="expressivity", hue="model", data=df, palette="Set2", style="model", markers=True, dashes=False, ci='sd')
    ax.set(ylim=(0, 1))
    ax.savefig("expressivity.png")
    plt.show()
    pdb_df = df.groupby(['temp', 'model']).describe()
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--models",
        type=list,
        nargs="+",
        default=[
            "TIMED_Deep",
            # "TIMED_not_so_deep_rot",
            "TIMED_rot",
            "TIMED_rotamer_deep_rot",
            "TIMED",
        ],
        help="Which models to analyse (default: TIMED_Deep, TIMED_not_so_deep_rot, TIMED_rot, TIMED_rotamer_deep_rot, TIMED).",
    )
    parser.add_argument(
        "--temperature",
        type=list,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0],
        help="Which Temperatures to analyse (default: 0.1, 0.5, 1.0, 2.0, 5.0).",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=200,
        help="Number of samples drawn from the distribution (default: 200).",
    )
    params = parser.parse_args()
    main(params)
