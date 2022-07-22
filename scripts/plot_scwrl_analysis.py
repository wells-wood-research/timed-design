import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ampal.amino_acids import standard_amino_acids

from design_utils.analyse_utils import tag_pdb_with_rot
from design_utils.utils import (
    get_rotamer_codec,
)

sns.set_theme(style="darkgrid")

def plot_scwrl_energy_scores(args):
    analysis_paths = list(args.input_path.glob('rotamer_analysis_*/scwrl_scores.csv'))
    a_path = pd.read_csv(analysis_paths[0])
    all_pdbs = a_path['PDB'].tolist()
    models = ['scwrl_real']
    values = [a_path['score_real'].tolist()]
    for a_path in analysis_paths:
        model_name = str(a_path).split("rotamer_analysis_")[1].split("/")[0]
        model_results = pd.read_csv(a_path)
        # Sanity check:
        assert model_results['PDB'].tolist() == all_pdbs, f"PDB Mismatch for model {a_path}"
        res = model_results['score_rot'].tolist()
        models.append(model_name)
        values.append(res)
    # Plot barplot + violin plot of energy scores
    analysis_df = pd.DataFrame(values, index=models).T
    g = sns.barplot(data=analysis_df, ci='sd', capsize=.2)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    plt.close()
    g = sns.violinplot(data=analysis_df, inner="quartile")
    plt.xticks(rotation=90)
    plt.ylim(-1500, 7000)
    plt.tight_layout()
    plt.show()

def plot_metrics(args):
    file_patterns = ['rotamer_analysis_*/results*_vs_scwrl*.txt', 'rotamer_analysis_*/results*_vs_wt.txt',
                     'rotamer_analysis_*/results*_vs_wt_scwrl.txt']
    pattern_names = ["Predicted Rotamers vs Predicted Residue + SCWRL", "Predicted Rotamers vs Wild Type Rotamers",
                     "Predicted Rotamers vs SCWRL of Wild Type"]
    for p_name, f_pattern in zip(pattern_names, file_patterns):
        rot_accuracy_files = list(args.input_path.glob(f_pattern))
        models = []
        values = {}
        for f in rot_accuracy_files:
            model_name = str(f).split("rotamer_analysis_")[1].split("/")[0]
            models.append(model_name)
            with open(f, "r") as current_f:
                for line in current_f.readlines():
                    metric, value = line.split(": ", 1)
                    # Sanitise Inputs:
                    metric = metric.split("Metrics ")[-1].lower()
                    if "report" in metric or "bias" in metric:

                        value = ast.literal_eval(str(value).replace("nan", "'nan'"))
                    else:
                        value = float(value) * 100
                    if metric in values:
                        values[metric].append(value)
                    else:
                        values[metric] = [value]

        for m in args.metrics_to_plot:
            m = m.lower()
            assert m in values.keys(), f"Metric {m} not found in {values.keys()}"
            analysis_df = pd.DataFrame(values[m], index=models).T
            ax = sns.barplot(data=analysis_df, capsize=.2)
            for container in ax.containers:
                ax.bar_label(container)
            plt.xticks(rotation=90)
            plt.ylim(0, 100)
            plt.ylabel(m + " (%)")
            plt.title(p_name)
            plt.tight_layout()
            plt.show()
            plt.close()

        if f_pattern == 'rotamer_analysis_*/results*_vs_wt.txt':
            analysis_paths = list(args.input_path.glob('rotamer_analysis_*/scwrl_scores.csv'))
            a_path = pd.read_csv(analysis_paths[0])
            all_pdbs = a_path['PDB'].tolist()

            rotamer_model_results_dict, _ = tag_pdb_with_rot(
                args.workers, args.path_to_pdb, all_pdbs
            )
            # Get rotamer categories:
            _, flat_categories = get_rotamer_codec()
            all_categories = []
            for rotamers in rotamer_model_results_dict.values():
                for n in rotamers:
                    # Deal with nan
                    if isinstance(n, int):
                        all_categories.append(flat_categories[n])
            categories, counts = np.unique(all_categories, return_counts=True)
            counts = (counts / np.sum(counts)) * 100
            predict_metrics = dict(zip(categories, counts))
            report_metrics = values['report']
            correlations = []
            for i, model in enumerate(models):
                curr_metrics = report_metrics[i]
                all_fscore = np.array([curr_metrics[res]['recall']*100 for res in predict_metrics.keys()])
                r = np.corrcoef(counts, all_fscore)
                correlations.append(r)

    raise ValueError

def main(args):
    args.input_path = Path(args.input_path)
    args.path_to_pdb = Path(args.path_to_pdb)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert args.path_to_pdb.exists(), f"PDB folder {args.path_to_pdb} does not exist"

    # Plot SCWRL scores
    # plot_scwrl_energy_scores(args)
    # Plot rotamer accuracy vs rotamers predicted sequence with scwrl
    plot_metrics(args)
    # Plot rotamer performance vs







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--metrics_to_plot",
        type=list,
        nargs="+",
        default=["accuracy", "accuracy_5"],
        help="Which metrics to plot. Default: accuracy",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    parser.add_argument(
        "--path_to_pdb",
        type=str,
        help="Path to biounit pdb dataset. Needs to be in format pdb/{2nd and 3rd char}/{pdb}.pdb1.gz",
    )
    params = parser.parse_args()
    main(params)
