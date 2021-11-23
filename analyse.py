import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ampal.amino_acids import standard_amino_acids


def plot_mean_var_probs(prediction_matrix, model_name):
    mean = np.mean(prediction_matrix, axis=1)
    var = np.var(prediction_matrix, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mean)
    axes[0].set_title("Mean")
    axes[0].set_xlim((np.min(mean), np.max(mean)))
    axes[1].hist(var)
    axes[1].set_title("Variance")
    axes[1].set_xlim((np.min(var), np.max(var)))
    plt.savefig(f"{model_name}_hist_var_mean.png")


def plot_sequence_heatmap(pdb_to_consensus_prob, model_name):
    sequences = ["1adnA", "1apoA", "1aq5A"]

    for seq in sequences:
        curr_prob = pdb_to_consensus_prob[seq]
        mean = np.mean(curr_prob, axis=1)
        var = np.var(curr_prob, axis=1)

        heatmap = sns.heatmap(mean, yticklabels=list(standard_amino_acids.keys()))
        heatmap.get_figure()
        heatmap.savefig(f"{model_name}_{seq}_mean_heatmap.png", dpi=400)
        heatmap = sns.heatmap(var, yticklabels=list(standard_amino_acids.keys()))
        heatmap.get_figure()
        heatmap.savefig(f"{model_name}_{seq}_var_heatmap.png", dpi=400)


def plot_patterns(pdb_to_consensus_prob, model_name):
    # Load prediction matrix
    prediction_matrix = np.genfromtxt(
        f"{model_name}.csv", delimiter=",", dtype=np.float16
    )
    plot_mean_var_probs(prediction_matrix, model_name)
    plot_sequence_heatmap(pdb_to_consensus_prob, model_name)

