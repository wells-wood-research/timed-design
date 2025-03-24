import typing as t
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import logomaker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from ampal.amino_acids import (
    bulkiness,
    standard_amino_acids,
    transmembrane_tendancy,
    uniprot_composition_2013,
)
from ampal.analyse_protein import (
    sequence_charge,
    sequence_isoelectric_point,
    sequence_molecular_weight,
)
from matplotlib.figure import Figure
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from tensorflow.python.keras.metrics import top_k_categorical_accuracy


def top_3_cat_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc

aa_to_index = {aa: i for i, aa in enumerate(standard_amino_acids.keys())}


class SequenceMetrics(NamedTuple):
    charge: float
    isoelectric_point: float
    molecular_weight: float
    composition_divergence: float
    bulkiness_score: float
    avg_tm_tendency: float


class DesignMetrics(NamedTuple):
    sequence_recovery: float
    overall_confidence: float
    avg_entropy: float
    position_entropy: t.List[float]


class SequenceMetricsCalculator:
    """
    Computes sequence metrics and returns a SequenceMetrics NamedTuple.
    """

    @staticmethod
    def calculate(seq) -> SequenceMetrics:
        """
        Calculate sequence metrics.

        Returns
        -------
        SequenceMetrics
        """
        if len(seq) == 0:
            raise ValueError(
                "Cannot calculate composition divergence for an empty sequence"
            )

        charge = sequence_charge(seq)
        iso_ph = sequence_isoelectric_point(seq)
        mw = sequence_molecular_weight(seq)
        divergence = SequenceMetricsCalculator._calculate_composition_divergence(seq)
        avg_bulk = SequenceMetricsCalculator._calculate_avg_property(seq, bulkiness)
        avg_tm = SequenceMetricsCalculator._calculate_avg_property(
            seq, transmembrane_tendancy
        )

        return SequenceMetrics(charge, iso_ph, mw, divergence, avg_bulk, avg_tm)

    @staticmethod
    def _calculate_avg_property(seq, property_dict: dict) -> float:
        """
        Computes the average property value over the sequence.

        Parameters
        ----------
        seq : str
            Amino acid sequence
        property_dict : dict
            Dict mapping one-letter amino acid codes to numerical values

        Returns
        -------
        float
            Average property value across valid amino acids in the sequence
        """
        values = [property_dict[aa] for aa in seq if aa in property_dict]
        if not values:
            raise ValueError("No valid amino acids in sequence for property")
        return sum(values) / len(values)

    @staticmethod
    def _calculate_composition_divergence(seq) -> float:
        """
        Calculate the mean absolute error between the sequence's amino acid composition
        and the UniProt 2013 amino acid composition.

        Parameters
        ----------
        seq : str
            Amino acid sequence

        Returns
        -------
        float
            Mean absolute error between normalized compositions
        """
        seq_len = len(seq)
        aa_counts = Counter(seq)
        seq_freq = {
            aa: (aa_counts.get(aa, 0) / seq_len) * 100
            for aa in uniprot_composition_2013
        }

        mae = sum(
            abs(seq_freq[aa] - uniprot_composition_2013[aa])
            for aa in uniprot_composition_2013
        ) / len(uniprot_composition_2013)

        return mae


class DesignedMetricsCalculator:
    """
    Computes sequence metrics and returns a DesignMetrics NamedTuple.
    """

    @staticmethod
    def calculate(
        seq: str,
        wildtype_seq: str,
        prob_matrix: np.ndarray,
        redesigned_indices: t.Optional[t.List[int]] = None,
    ) -> DesignMetrics:
        if len(seq) == 0:
            raise ValueError(
                "Cannot calculate composition divergence for an empty sequence"
            )
        if redesigned_indices is None:
            redesigned_indices = list(range(len(seq)))

        sequence_recovery = DesignedMetricsCalculator._calculate_sequence_recovery(
            seq, wildtype_seq, redesigned_indices
        )
        overall_conf = DesignedMetricsCalculator.calculate_overall_confidence(
            seq, prob_matrix, redesigned_indices
        )
        (
            entropy_arr,
            avg_entropy,
        ) = DesignedMetricsCalculator._calculate_prediction_entropy(prob_matrix)

        return DesignMetrics(
            sequence_recovery=sequence_recovery,
            overall_confidence=overall_conf,
            avg_entropy=avg_entropy,
            position_entropy=entropy_arr.tolist(),
        )

    @staticmethod
    def _calculate_sequence_recovery(
        seq: str, wildtype_seq: str, indices: t.List[int]
    ) -> float:
        """
        Calculates percentage identity between designed and wild-type sequences over given indices.
        """
        matches = sum(1 for i in indices if seq[i] == wildtype_seq[i])
        return matches / len(indices) if indices else float("nan")

    @staticmethod
    def _calculate_prediction_entropy(
        residue_predictions: np.ndarray,
    ) -> t.Tuple[t.List[float], float]:
        """
        Calculates Shannon Entropy on predictions. From the TIMED repository.

        Parameters
        ----------
        residue_predictions: np.ndarray[float]
            Residue probabilities for each position in sequence of shape (n, 20)
            where n is the number of residues in sequence.

        Returns
        -------
        entropy_arr: np.ndarray
            Entropy of prediction for each position in sequence of shape (n,).
        entropy_mean: float
            Overall mean entropy of predictions.
        """
        entropy_arr = entropy(residue_predictions, base=2, axis=1)
        return entropy_arr, float(np.mean(entropy_arr))

    @staticmethod
    def calculate_overall_confidence(
        selected_sequence: str,
        prob_matrix: np.ndarray,
        redesigned_indices: t.List[int],
        aa_to_index: t.Dict[str, int],
    ) -> float:
        """
        Calculates overall confidence: exp(mean log-probability over redesigned positions).

        Parameters
        ----------
        selected_sequence: str
            Designed sequence
        prob_matrix: np.ndarray[float]
            Residue probabilities for each position in sequence of shape (n, 20)
            where n is the number of residues in sequence.
        redesigned_indices: t.List[int]
            List of indices that were redesigned
        aa_to_index: t.Dict[str, int]
            Mapping from amino acid to index in prob_matrix

        """
        log_probs = []
        for i in redesigned_indices:
            aa = selected_sequence[i]
            if aa not in aa_to_index:
                raise ValueError(f"Invalid amino acid '{aa}' at position {i}")
            prob = prob_matrix[i, aa_to_index[aa]]
            log_probs.append(np.log(prob + 1e-12))  # prevent log(0)

        return float(np.exp(np.mean(log_probs)))


def create_sequence_logo(prediction_matrix: np.ndarray) -> Figure:
    """
    Create sequence logo for prediction matrix

    Parameters
    ----------
    prediction_matrix: np.ndarray
        Prediction matrix (n, 20) or (n,388)

    Returns
    -------
    fig: Figure
        Matplotlib fig of sequence logo

    """
    prediction_df = pd.DataFrame(
        prediction_matrix, columns=list(standard_amino_acids.keys())
    )
    # create Logo object
    seq_logo = logomaker.Logo(
        prediction_df,
        color_scheme="chemistry",
        vpad=0.1,
        width=0.8,
        figsize=(
            max(0.12 * len(prediction_matrix), 10),
            max(0.03 ** len(prediction_matrix), 2.5),
        ),
    )
    seq_logo.style_xticks(anchor=0, spacing=5)
    seq_logo.ax.set_ylabel("Probability (%)")
    seq_logo.ax.set_xlabel("Residue Position")
    return seq_logo.ax.get_figure()


def plot_confusion_matrix(
    cm: np.ndarray,
    y_labels: t.List[str],
    x_labels: t.List[str],
    title: str,
    output_path: Path,
    display_colorbar: bool = False,
):
    """
    Plot confusion matrix (can be any shape) to a file

    Parameters
    ----------
    cm: np.ndarray
        Confusion matrix of shape (len(y_labels), len(x_labels))
    y_labels: t.List[str]
        List of string of y labels
    x_labels: t.List[str]
        List of string of x labels
    title: str
        Title string for the graph (will be used as filename without spaces)
    display_colorbar:
        Whether to display the colorbar on the right hand side of the graph
    """
    # Plot Confusion Matrix:
    fig = plt.figure(figsize=(max(len(x_labels) * 0.5, 5), max(len(y_labels) * 0.5, 5)))
    # fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.xlabel("Predicted Residue")
    plt.xticks(range(len(x_labels)), x_labels, rotation=90)
    plt.ylabel("True Residue")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(f"{title}")
    # Plot Color Bar:
    norm = colors.Normalize()
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    if display_colorbar:
        fig.colorbar(sm).set_label("Confusion Level (Range 0 - 1)")
    fig.tight_layout()
    fig.savefig(output_path / f"{title.replace(' ', '_')}.png")
    # Save Confusion:
    plt.close()


def calculate_ui_metrics(pdb_to_sequence: dict, pdb_to_real_sequence: dict) -> dict:
    """
    Calculates useful metrics for sequence performance analysis. Metrics calculated are:
        - Classification report (https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
        - Accuracy (top 1, 2, 3, 4, 5 accuracy)
        - Macro Precision and Recall
        - Prediction bias (https://developers.google.com/machine-learning/crash-course/classification/prediction-bias)
        - Weighted / Unweighted confusion matrix
        - Count of each residue in dataset
        - Count of each residue in predictions


    Parameters
    ----------
    pdb_to_sequence: dict
        Dictionary {pdb_code: sequence}
    pdb_to_real_sequence: dict
        Dictionary {pdb_code: real_sequence}

    Returns
    -------
    metrics: dict
        Dictionary of metrics

    """
    y_pred, y_true = encode_sequence_to_onehot(pdb_to_sequence, pdb_to_real_sequence)
    y_pred_argmax = np.argmax(y_pred, axis=1)
    y_true_argmax = np.argmax(y_true, axis=1)
    flat_categories = list(standard_amino_acids.keys())
    report = classification_report(
        y_pred_argmax,
        y_true_argmax,
        labels=list(range(len(flat_categories))),
        target_names=flat_categories,
        output_dict=True,  # Returns a dictionary
    )
    accuracy_1 = accuracy_score(y_true_argmax, y_pred_argmax)
    accuracy_2 = top_k_accuracy_score(
        y_true_argmax, y_pred, k=2, labels=list(range(len(flat_categories)))
    )
    accuracy_3 = top_k_accuracy_score(
        y_true_argmax, y_pred, k=3, labels=list(range(len(flat_categories)))
    )
    accuracy_4 = top_k_accuracy_score(
        y_true_argmax, y_pred, k=4, labels=list(range(len(flat_categories)))
    )
    accuracy_5 = top_k_accuracy_score(
        y_true_argmax, y_pred, k=5, labels=list(range(len(flat_categories)))
    )
    precision = precision_score(
        y_pred_argmax,
        y_true_argmax,
        average="macro",
        labels=list(range(len(flat_categories))),
        zero_division=0,
    )
    recall = recall_score(
        y_pred_argmax,
        y_true_argmax,
        average="macro",
        labels=list(range(len(flat_categories))),
        zero_division=0,
    )
    # Calculate bias:
    count_labels = Counter(y_true_argmax)
    count_pred = Counter(y_pred_argmax)
    bias = {}
    sum_counts = len(y_true)
    for y, _ in enumerate(standard_amino_acids.keys()):
        if y in count_labels:
            c_label = count_labels[int(y)] / sum_counts
        else:
            c_label = 0
        if y in count_pred:
            c_pred = count_pred[int(y)] / sum_counts
        else:
            c_pred = 0
        b = c_pred - c_label
        bias[flat_categories[int(y)]] = b

    unweighted_cm = confusion_matrix(
        y_true_argmax,
        y_pred_argmax,
        normalize="all",
        labels=list(range(len(standard_amino_acids.keys()))),
    )

    return {
        "report": report,
        "accuracy_1": accuracy_1,
        "accuracy_2": accuracy_2,
        "accuracy_3": accuracy_3,
        "accuracy_4": accuracy_4,
        "accuracy_5": accuracy_5,
        "precision": precision,
        "recall": recall,
        "count_labels": count_labels,
        "count_pred": count_pred,
        "bias": bias,
        "unweighted_cm": unweighted_cm,
    }


def encode_sequence_to_onehot(pdb_to_sequence: dict, pdb_to_real_sequence: dict):
    y_pred = []
    y_true = []
    one_hot_encode = np.zeros((len(standard_amino_acids), len(standard_amino_acids)))
    diag = np.arange(len(standard_amino_acids))
    one_hot_encode[diag, diag] = 1
    r_num = dict(zip(standard_amino_acids.keys(), one_hot_encode))
    # Extract predictions:
    for pdb in pdb_to_sequence.keys():
        if pdb in pdb_to_real_sequence:
            current_true = []
            current_pred = []
            for r_t, r_p in zip(pdb_to_real_sequence[pdb], pdb_to_sequence[pdb]):
                current_true.append(r_num[r_t])
                current_pred.append(r_num[r_p])
            y_true += current_true
            y_pred += current_pred
        else:
            print(f"Error with pdb code {pdb}")
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return y_pred, y_true
