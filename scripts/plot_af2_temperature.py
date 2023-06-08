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
        "ranked_0_rmsd",
        "ranked_0_gdt",
        "ranked_1_rmsd",
        "ranked_1_gdt",
        "ranked_2_rmsd",
        "ranked_2_gdt",
        "ranked_3_rmsd",
        "ranked_3_gdt",
        "ranked_4_rmsd",
        "ranked_4_gdt",
        "relaxed_model_1_rmsd",
        "relaxed_model_1_gdt",
        "relaxed_model_2_rmsd",
        "relaxed_model_2_gdt",
        "relaxed_model_3_rmsd",
        "relaxed_model_3_gdt",
        "relaxed_model_4_rmsd",
        "relaxed_model_4_gdt",
        "relaxed_model_5_rmsd",
        "relaxed_model_5_gdt",
        "unrelaxed_model_1_rmsd",
        "unrelaxed_model_1_gdt",
        "unrelaxed_model_2_rmsd",
        "unrelaxed_model_2_gdt",
        "unrelaxed_model_3_rmsd",
        "unrelaxed_model_3_gdt",
        "unrelaxed_model_4_rmsd",
        "unrelaxed_model_4_gdt",
        "unrelaxed_model_5_rmsd",
        "unrelaxed_model_5_gdt",
    ]
    # Load results in pandas
    df = pd.read_csv(args.input_path)
    df.columns = columns
    df = df.replace(0.0, np.nan)
    # df = df.groupby(["model"])
    for name in columns[4:]:
        fig, axs = plt.subplots(ncols=2, figsize=(10,5))
        # Set Palette:
        p = sns.color_palette("Set2")
        palette = {
            "TIMED_rot": p[0],
            "TIMED": p[1],
        }
        metric = name.split("_")[-1].upper()
        # Graph Solubility and Expressivity:
        sns.lineplot(
            x="temperature",
            y=name,
            hue="model",
            data=df,
            palette=palette,
            style="model",
            markers=True,
            dashes=False,
            ax=axs[0],
            legend=False,
        )
        axs[0].set(ylabel=metric, title=f"{metric}")
        # axs[0].figure.savefig(f"lineplot_{name}_{metric}.png")
        df2 = df.groupby(["temperature", "model"]).describe()
        sns.lineplot(
            x="temperature",
            y=(name, "std"),
            hue="model",
            data=df2,
            palette=palette,
            style="model",
            markers=True,
            dashes=False,
            ax=axs[1],
        )
        axs[1].set(ylabel=f"STDev on {metric}", title=f"STDev of {metric}")
        # axs[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(f"{metric}_{name}.png")
        plt.show()
        plt.close()
    # plt.close()
    # ax = sns.lineplot(
    #     x="temp",
    #     y="expressivity",
    #     hue="model",
    #     data=df,
    #     palette="Set2",
    #     style="model",
    #     markers=True,
    #     dashes=False,
    # )
    # ax.set(ylim=(0, 0.5), ylabel="MAE (expressivity)")
    # ax.figure.savefig("expressivity_error.png")
    # plt.close()
    # # Graph Error on Sol and Exp:
    # df = df.groupby(["temp", "model"]).describe()
    # ax = sns.lineplot(
    #     x="temp",
    #     y=("solubility", "std"),
    #     hue="model",
    #     data=df,
    #     palette="Set2",
    #     style="model",
    #     markers=True,
    #     dashes=False,
    # )
    # ax.set(ylim=(0, 0.5), ylabel="STDev on MAE (solubility)")
    # ax.figure.savefig("solubility_var.png")
    # plt.close()
    # ax = sns.lineplot(
    #     x="temp",
    #     y=("expressivity", "std"),
    #     hue="model",
    #     data=df,
    #     palette="Set2",
    #     style="model",
    #     markers=True,
    #     dashes=False,
    # )
    # ax.set(ylim=(0, 0.5), ylabel="STDev on MAE (expressivity)")
    # ax.figure.savefig("expressivity_var.png")
    # plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    params = parser.parse_args()
    main(params)
