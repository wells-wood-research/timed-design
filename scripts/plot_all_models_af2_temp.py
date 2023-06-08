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
    columns = [
        "model",
        "pdb",
        "n",
        "temperature",
        "accuracy",
        "rmsd",
        "gdt",
    ]
    for curr_model in args.models:
        curr_model_path = args.input_path / f"all_results_{curr_model}.csv"
        assert curr_model_path.exists(), f"Input file {curr_model_path} does not exist"
        # Load results in pandas:
        df = pd.read_csv(curr_model_path, header=None)
        df.dropna(inplace=True)
        results_arr += df.values.tolist()
    # Merge all together:
    results_arr = pd.DataFrame(results_arr, columns=columns)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    # Set Palette:
    p = sns.color_palette("Set2")
    palette = {
        "TIMED-Deep": p[0],
        "TIMED": p[1],
        "TIMED-rotamer-deep-rot": p[2],
        "TIMED-rotamer-rot": p[3],
    }
    # Graph Solubility and Expressivity:
    sns.lineplot(
        x="temperature",
        y="rmsd",
        hue="model",
        data=results_arr,
        palette=palette,
        style="model",
        markers=True,
        dashes=False,
        ax=axs[0],
        legend=False,
    )
    # axs[0].set(ylabel=f"ac", title=f"a")
    df2 = results_arr.groupby(["temperature", "model"]).describe()
    sns.lineplot(
        x="temperature",
        y=("rmsd", "std"),
        hue="model",
        data=df2,
        palette=palette,
        style="model",
        markers=True,
        dashes=False,
        ax=axs[1],
    )
    axs[1].set(ylabel=f"STDev on RMSD", title=f"Standard Deviation of RMSD at Different Temperatures")
    axs[0].set(ylabel=r"RMSD $\AA$", title=f"RMSD at Different Temperatures")
    plt.tight_layout()
    plt.savefig('rmsd_std.png')
    plt.close()
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    # Set Palette:
    p = sns.color_palette("Set2")
    palette = {
        "TIMED-Deep": p[0],
        "TIMED": p[1],
        "TIMED-rotamer-deep-rot": p[2],
        "TIMED-rotamer-rot": p[3],
    }
    # Graph Solubility and Expressivity:
    sns.lineplot(
        x="temperature",
        y="accuracy",
        hue="model",
        data=results_arr,
        palette=palette,
        style="model",
        markers=True,
        dashes=False,
        ax=axs[0],
        legend=False,
    )
    df2 = results_arr.groupby(["temperature", "model"]).describe()
    sns.lineplot(
        x="temperature",
        y=("accuracy", "std"),
        hue="model",
        data=df2,
        palette=palette,
        style="model",
        markers=True,
        dashes=False,
        ax=axs[1],
    )
    axs[1].set(ylabel=f"STDev on Accuracy", title=f"Standard Deviation of Accuracy at Different Temperatures")
    axs[0].set(ylabel=f"Accuracy (%)", title=f"Accuracy at Different Temperatures")
    plt.tight_layout()
    plt.savefig('accuracy_std.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--models",
        type=list,
        nargs="+",
        default=[
            "TIMED-Deep",
            "TIMED-rotamer-deep-rot",
            "TIMED-rotamer-rot",
            "TIMED",
        ],
        help="Which models to analyse (default: all).",
    )
    params = parser.parse_args()
    main(params)
