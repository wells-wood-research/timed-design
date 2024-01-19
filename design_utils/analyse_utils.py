import gzip
import typing as t
from collections import Counter
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import ampal
import logomaker
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ampal.amino_acids import standard_amino_acids
from ampal.analyse_protein import (
    sequence_charge,
    sequence_isoelectric_point,
    sequence_molar_extinction_280,
    sequence_molecular_weight,
)
from scipy.stats import entropy
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from aposteriori.data_prep.create_frame_data_set import _fetch_pdb
from design_utils.scwrl_utils import pack_side_chains_scwrl
from design_utils.utils import (
    get_rotamer_codec,
    load_datasetmap,
    extract_sequence_from_pred_matrix,
)
from design_utils.utils import compress_rotamer_predictions_to_20

# input type is either ampal.Assembly or ampal.Polypeptide
def tag_packing_density(
    structure: t.Union[ampal.Polymer, ampal.Assembly], radius: float = 7
) -> None:
    """
    Function from ISAMBARD written by 'Kathryn L. Shelley'

    Calculates the packing density of each non-hydrogen atom in a polymer
    or assembly.

    An atom's packing density is a measure of the number of atoms within
    its local environment. There are several different methods of
    calculating packing density; we use atomic contact number [1], which
    is the number of non-hydrogen atoms within a specified radius (default
    7 A [1]).

    Parameters
    ----------
    structure : ampal.Polymer or ampal.Assembly
        The structure to be tagged.
    radius : float, optional
        The radius (in Angstroms) within which to count atoms. Default is 7 A.

    References
    ----------
    .. [1] Weiss MS (2007) On the interrelationship between atomic
       displacement parameters (ADPs) and coordinates in protein
       structures. *Acta Cryst.* D**63**, 1235-1242.
    """

    if not type(structure).__name__ in ["Polymer", "Assembly"]:
        raise ValueError(
            "Contact order can only be calculated for a polymer or an assembly."
        )

    atoms_list = [atom for atom in list(structure.get_atoms()) if atom.element != "H"]
    atom_coords_array = np.array([atom.array for atom in atoms_list])

    for index, atom in enumerate(atoms_list):
        distances = np.sqrt(
            np.square(atom_coords_array[:, :] - atom_coords_array[index, :]).sum(axis=1)
        )
        # Subtract 1 to correct for the atom itself being counted
        atom.tags["packing density"] = np.sum(distances < radius) - 1


def _extract_bfactor_from_polypeptide(assembly: ampal.Polypeptide) -> t.List[float]:
    """
    Extracts bfactor from ampal polypeptide

    Parameters
    ----------
    assembly: ampal.Polypeptide
        Ampal polypeptide to extract bfactor from.

    Returns
    -------

    """
    bfactors = []
    # Extract iddt for each residue
    for res in assembly:
        # All the atoms have the same bfactor (iddt) so select first atom:
        first_atom = list(res.atoms.keys())[0]
        curr_iddt = res.atoms[first_atom].tags["bfactor"]
        bfactors.append(curr_iddt)
    return bfactors


def extract_bfactor_from_ampal(pdb_path: Path, load_pdb: bool = True) -> t.List[float]:
    """
    Extracts bfactor from ampal assembly or polypeptide

    Parameters
    ----------
    pdb_path: Path
        Path to pdb file or ampal assembly
    load_pdb: bool
        Whether to load pdb file or not

    Returns
    -------
    all_b_factors: t.List[float]
        List of bfactor for each residue in assembly or polypeptide

    """
    all_b_factors = []
    if load_pdb:
        assembly = ampal.load_pdb(pdb_path)
    else:
        assembly = pdb_path

    if isinstance(assembly, ampal.AmpalContainer):
        assembly = assembly[0]
    if isinstance(assembly, ampal.Assembly):
        for assem in assembly:
            if isinstance(assem, ampal.Polypeptide):
                bfactors = _extract_bfactor_from_polypeptide(assem)
                all_b_factors.append(bfactors)
    elif isinstance(assembly, ampal.Polypeptide):
        bfactors = _extract_bfactor_from_polypeptide(assembly)
        all_b_factors.append(bfactors)

    return all_b_factors


def _extract_packdensity_from_polypeptide(assembly: ampal.Assembly, atom_filter: str) -> t.List[float]:
    """
    Extracts packing density from ampal polypeptide

    Parameters
    ----------
    assembly: ampal.Assembly
        Ampal assembly to extract packing density from.
    atom_filter: str
        Atom filter function to use. Can be "backbone", "ca" or "all"

    Returns
    -------
    packdensity: t.List[float]
        List of packing density for each residue in assembly or polypeptide
    """
    if atom_filter == "backbone":
        filter_set = ("N", "CA", "C", "O")
    elif atom_filter == "ca":
        filter_set = "CA"
    elif atom_filter == "all":
        filter_set = None
    else:
        raise ValueError(
            f"Atom Filter function {atom_filter} not in (backbone, ca, all)"
        )

    packdensity = []
    tag_packing_density(assembly)
    # Extract iddt for each residue
    for res in assembly[0]:
        # All the atoms have the same bfactor (iddt) so select first atom:
        current_density = -1
        for atom in res:
            if filter_set:
                if atom.res_label in filter_set:  # Only backbone atoms
                    if current_density == -1:
                        current_density = atom.tags["packing density"]
                    else:
                        current_density = (
                            current_density + atom.tags["packing density"]
                        ) / 2
            else:
                if atom.res_label != "H":
                    if current_density == -1:
                        current_density = atom.tags["packing density"]
                    else:
                        current_density = (
                            current_density + atom.tags["packing density"]
                        ) / 2

        packdensity.append(current_density)
    return packdensity


def extract_packdensity_from_ampal(pdb: t.Union[str, Path], load_pdb: bool =True, atom_filter: str = "ca") -> t.List[float]:
    """
    Extracts packing density from ampal assembly or polypeptide

    Parameters
    ----------
    pdb: t.Union[str, Path]
        Path to pdb file or pdb string
    load_pdb: bool
        Whether to load pdb file or not
    atom_filter: str
        Atom filter function to use. Can be "backbone", "ca" or "all"

    Returns
    -------
    all_packdensity: t.List[float]
        List of packing density for each residue in assembly or polypeptide

    """
    all_packdensity = []
    if load_pdb:
        assembly = ampal.load_pdb(pdb)
    else:
        assembly = pdb
    if isinstance(assembly, ampal.AmpalContainer):
        assembly = assembly[0]
    if isinstance(assembly, ampal.Assembly):
        packdensity = _extract_packdensity_from_polypeptide(assembly, atom_filter)
        all_packdensity.append(packdensity)

    return all_packdensity


def extract_prediction_entropy_to_dict(
    model_pred_path: Path, model_map_path: Path, rotamer_mode: bool=False, is_old: bool=False
) -> dict:
    """
    Extracts prediction entropy from prediction matrix and datasetmap.

    Parameters
    ----------
    model_pred_path: Path
        Path to prediction matrix
    model_map_path: Path
        Path to datasetmap
    rotamer_mode: bool
        Whether to use rotamer mode or not
    is_old: bool
        Whether to use old datasetmap or not

    Returns
    -------
    pdb_to_entropy: dict
        Dictionary {pdb_code: entropy}
    """
    assert model_pred_path.exists(), f"Model path {model_pred_path} does not exists."
    assert model_map_path.exists(), f"Model path {model_map_path} does not exists."
    # Load prediction matrix:
    prediction_matrix = np.genfromtxt(model_pred_path, delimiter=",", dtype=np.float64)
    # Load datasetmap
    datasetmap = load_datasetmap(model_map_path, is_old=is_old)
    if rotamer_mode:
        # Get rotamer categories:
        _, flat_categories = get_rotamer_codec()
        # Get dictionary for 3 letter -> 1 letter conversion:
        res_to_r = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
        # Create flat categories of 1 letter amino acid for each of the 338 rotamers:
        flat_categories = [res_to_r[res.split("_")[0]] for res in flat_categories]
        # Extract dictionaries with sequences:
    else:
        _, flat_categories = None, None
    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        _,
        _,
    ) = extract_sequence_from_pred_matrix(
        datasetmap,
        prediction_matrix,
        rotamers_categories=flat_categories,
        old_datasetmap=is_old,
    )
    pdb_to_entropy = {}
    for pdb, prob in pdb_to_probability.items():
        curr_entropy = calculate_prediction_entropy(prob)
        pdb_to_entropy[pdb] = curr_entropy
    return pdb_to_entropy


def calculate_prediction_entropy(residue_predictions: t.List[float]) -> t.List[float]:
    """
    Calculates Shannon Entropy on predictions. From the TIMED repository.

    Parameters
    ----------
    residue_predictions: list[float]
        Residue probabilities for each position in sequence of shape (n, 20)
        where n is the number of residues in sequence.

    Returns
    -------
    entropy_arr: list[float]
        Entropy of prediction for each position in sequence of shape (n,).
    """
    entropy_arr = entropy(residue_predictions, base=2, axis=1)
    return entropy_arr


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
    if prediction_matrix.shape[-1] == 338:
        prediction_matrix = compress_rotamer_predictions_to_20(prediction_matrix)

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


def calculate_seq_metrics(seq: str) -> t.Tuple[float, float, float, float]:
    """
    Calculates sequence metrics.

    Currently only supports: Charge at pH 7, Isoelectric Point, Molecular Weight

    Parameters
    ----------
    seq: str
        Sequence of residues

    Returns
    -------
    metrics: t.Tuple[float, float, float, float]
        (charge , iso_ph, mw, me)
    """
    charge = sequence_charge(seq)
    iso_ph = sequence_isoelectric_point(seq)
    mw = sequence_molecular_weight(seq)
    me = sequence_molar_extinction_280(seq)
    return charge, iso_ph, mw, me


def save_assembly_to_path(structure: ampal.Assembly, output_dir: Path, name: str) -> None:
    """
    Saves ampal assembly to specified path.

    Parameters
    ----------
    structure: ampal.Assembly
        Ampal assembly to be saved
    output_dir: Path
        Output Directory
    name: str
        Name of output File
    """
    # Save assembly to path:
    output_path = output_dir / (name + ".pdb")
    with open(output_path, "w") as f:
        f.write(structure.pdb)


def pack_sidechains(
    structure: ampal.Assembly, sequence: str, scwrl_path: Path
) -> ampal.Assembly:
    """
    Packs sequence of residues onto ampal assembly using SCWRL

    Parameters
    ----------
    structure: ampal.Assembly
        Ampal assembly to be saved
    sequence: str
        Sequence of amino acids

    Returns
    -------
    packed_structure: ampal.Assembly
        Packed structure with scwrl
    """
    return pack_side_chains_scwrl(
        assembly=structure,
        sequences=sequence,
        rigid_rotamer_model=False,
        scwrl_path=scwrl_path,
    )


def analyse_with_scwrl(
    pdb_to_seq: dict,
    pdb_to_assembly: dict,
    output_path: Path,
    suffix: str,
    scwrl_path: Path,
) -> (dict, dict):
    """
    Analyses rotamer prediction with SCWRL

    Parameters
    ----------
    pdb_to_seq: dict
        {pdb_code: sequence}
    pdb_to_assembly:
        {pdb_code: ampal_assembly}
    output_path: Path
        Path to save analysis to.
    suffix: str
        Additional information to add to file.

    Returns
    -------
    pdb_to_scores: dict
        Dict {pdb_code: scwrl_score}
    pdb_to_errors: dict
         Dict {pdb_code: Error}
    """
    pdb_to_scores = {}
    pdb_to_errors = {}
    # Loop through each PDB code and pack them with SCWRL:
    for pdb in tqdm(
        pdb_to_seq.keys(), desc=f"Packing sequence in PDB {suffix} with SCWRL"
    ):
        pdb_outpath = output_path / (pdb + "_" + suffix + ".pdb")
        if pdb_outpath.exists():
            error = f"PDB {pdb} at {pdb_outpath} already exists."
            pdb_to_errors[pdb] = error
        elif pdb[:4] in pdb_to_assembly.keys():
            try:
                # If there are more than one backbones, add their sequences up for SCWRL:
                if len(pdb_to_assembly[pdb[:4]].backbone) > 1:
                    pdb_to_seq[pdb] = [pdb_to_seq[pdb]] * len(pdb_to_assembly[pdb[:4]])
                # Else structure is already monomeric - no need to add sequences
                else:
                    pdb_to_seq[pdb] = [
                        pdb_to_seq[pdb]
                    ]  # Sequences need to be in list for SCWRL4
                # Attempt packing:
                try:
                    scwrl_structure = pack_sidechains(
                        pdb_to_assembly[pdb[:4]], pdb_to_seq[pdb], scwrl_path=scwrl_path
                    )
                    pdb_to_scores[pdb] = scwrl_structure.tags["scwrl_score"]
                    save_assembly_to_path(
                        structure=scwrl_structure,
                        output_dir=output_path,
                        name=pdb + suffix,
                    )
                except ValueError as e:
                    error = f"Attempted packing on structure {pdb}, but got {e}"
                    pdb_to_errors[pdb] = error
            except (ValueError, KeyError) as e:
                error = f"Attempted selecting backbone on structure {pdb}, but got {e}"
                pdb_to_errors[pdb] = error
            except ChildProcessError as e:
                error = f"Attempted selecting backbone on structure {pdb}, but SCWRL failed: {e}"
                pdb_to_errors[pdb] = error
        else:
            error = f"Error with structure {pdb}. Assembly not found."
            pdb_to_errors[pdb] = error
    # Saves errors to file:
    output_error_path = output_path / f"errors_scwrl{suffix}.csv"
    print(
        f"Got {len(pdb_to_errors)} errors when attempting to pack {len(pdb_to_seq)} sequences. Saved errors in file {output_error_path}"
    )
    with open(output_error_path, "w") as f:
        for pdb, err in pdb_to_errors.items():
            f.write(f"{pdb},{err}\n")
    return pdb_to_scores, pdb_to_errors


def plot_cm(
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


def create_rot_cm(
    cm: np.ndarray, rot_categories: t.List[str], mode: str, output_path: Path
):
    """
    Create rotamer confusion matrices.

    Parameters
    ----------
    cm: np.ndarray
        (338, 338) Ci,j is equal to the number of observations known to be in group i and predicted to be in group j.
    rot_categories: t.List[str]
        List of strings of rotamer categories eg [ALA_0, CYS_ ...] (338,)
    mode: str
        Additional text preceding file name (eg. weighted_{filename}.svg)
    """
    rot_categories = np.asarray(rot_categories)
    # Get repeated residue labels for each residue (338,)
    res_categories = np.array([res.split("_")[0] for res in rot_categories])
    for res in standard_amino_acids.values():
        rot_idx = res_categories == res
        curr_rot_cat = rot_categories[rot_idx]
        # Select from CM
        rot_cm = cm[rot_idx, :]  # (n_rot, 338)
        rot_cm = rot_cm / np.sum(rot_cm)  # (n_rot, 338)
        small_cm = cm[rot_idx][:, rot_idx]  # (n_rot, n_rot)
        small_cm = small_cm / np.sum(rot_cm)  # (n_rot, n_rot)
        # Plot CM
        plot_cm(
            rot_cm,
            y_labels=curr_rot_cat,
            x_labels=rot_categories,
            title=f"{mode} {res} vs all 338 rot",
            output_path=output_path,
        )
        if len(small_cm) > 1:  # Avoids bug with glycine and alanine
            plot_cm(
                small_cm,
                y_labels=curr_rot_cat,
                x_labels=curr_rot_cat,
                title=f"{mode} {res} vs {res} rot",
                output_path=output_path,
            )
        rot_res_cm = np.zeros(
            (sum(rot_idx), 20)
        )  # (n_rot, 20) 1 extra column added for sum
        for i, r in enumerate(standard_amino_acids.values()):
            curr_rot_idx = res_categories == r
            curr_sum = np.sum(rot_cm[:, curr_rot_idx], axis=1)  # (n_rot, 1)
            rot_res_cm[:, i] = curr_sum
        rot_res_cm = rot_res_cm / np.sum(rot_res_cm)
        plot_cm(
            rot_res_cm,
            y_labels=curr_rot_cat,
            x_labels=list(standard_amino_acids.values()),
            title=f"{mode} {res} vs 20 res",
            output_path=output_path,
        )


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


def calculate_metrics(pdb_to_sequence: dict, pdb_to_real_sequence: dict) -> dict:
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


def calculate_rotamer_metrics(
    pdb_to_probability: dict,
    pdb_to_rotamer: dict,
    rot_categories: t.List[str],
    suffix: str,
    output_path: Path,
) -> None:
    """
    Calculates useful metrics for rotamer performance analysis:
        - ROC AUC score (OVO)
        - Classification report (https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
        - Accuracy (top 1, 2, 3, 4, 5 accuracy)
        - Macro Precision and Recall
        - Prediction bias (https://developers.google.com/machine-learning/crash-course/classification/prediction-bias)
        - Weighted / Unweighted confusion matrix

    Parameters
    ----------
    pdb_to_probability: dict
        Dictionary {pdb_code: probability}
    pdb_to_rotamer: dict
        Dictionary {pdb_code: rotamer_list}
    rot_categories: t.List[str]
        List of strings of rotamer categories eg [ALA_0, CYS_ ...] (338,)
    """
    y_pred = []
    y_true = []
    # Extract predictions:
    for pdb in pdb_to_probability.keys():
        if pdb in pdb_to_rotamer:
            if len(pdb_to_probability[pdb]) == len(pdb_to_rotamer[pdb]):
                y_pred += pdb_to_probability[pdb]
                y_true += pdb_to_rotamer[pdb]
            else:
                print(f"Error with pdb code {pdb} - Length Mismatch")
        else:
            print(f"Error with pdb code {pdb}")
    y_pred = np.array(y_pred).reshape(-1, 338)
    y_true = np.array(y_true).flatten()
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    # Sometimes predictions miss a very small amount due to the way they are saved:
    if not np.allclose(1, y_pred.sum(axis=1)):
        idx_res = y_pred.sum(axis=1) != 1
        old_residuals = y_pred[idx_res].sum(axis=1)
        # Calculate residual and then create idx_res, 338 with the residual / 338
        residual = (
            np.ones((338, idx_res.sum())) * (1 - y_pred[idx_res].sum(axis=1)) / 338
        )
        y_pred[idx_res] = np.add(y_pred[idx_res], residual.T)
        assert np.allclose(
            1, y_pred.sum(axis=1)
        ), f"Probabilities at idx {idx_res} do not add up to 1: {old_residuals} and after adjustment got {y_pred[idx_res]}"
    y_argmax = np.argmax(y_pred, axis=1)
    # Calculate metrics:
    auc_ovo = roc_auc_score(
        y_true,
        y_pred,
        multi_class="ovo",
        labels=list(range(len(rot_categories))),
        average="macro",
    )
    try:
        auc_ovr = roc_auc_score(
            y_true,
            y_pred,
            multi_class="ovr",
            labels=list(range(len(rot_categories))),
            average="macro",
        )
    except:
        auc_ovr = np.nan
    report = classification_report(
        y_true,
        y_argmax,
        labels=list(range(len(rot_categories))),
        target_names=rot_categories,
        output_dict=True,  # Returns a dictionary
    )
    accuracy = accuracy_score(y_true, y_argmax)
    accuracy_2 = top_k_accuracy_score(
        y_true, y_pred, k=2, labels=list(range(len(rot_categories)))
    )
    accuracy_3 = top_k_accuracy_score(
        y_true, y_pred, k=3, labels=list(range(len(rot_categories)))
    )
    accuracy_4 = top_k_accuracy_score(
        y_true, y_pred, k=4, labels=list(range(len(rot_categories)))
    )
    accuracy_5 = top_k_accuracy_score(
        y_true, y_pred, k=5, labels=list(range(len(rot_categories)))
    )
    precision = precision_score(
        y_true,
        y_argmax,
        average="macro",
        labels=list(range(len(rot_categories))),
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_argmax,
        average="macro",
        labels=list(range(len(rot_categories))),
        zero_division=0,
    )
    print("Metrics AUC_OVR")
    print(auc_ovr)
    print("Metrics AUC_OVO")
    print(auc_ovo)
    print("Metrics Report")
    print(report)
    print(
        f"Accuracy: {accuracy}, accuracy_2: {accuracy_2}, accuracy_3: {accuracy_3}, accuracy_4: {accuracy_4}, accuracy_5: {accuracy_5}, precision: {precision}, recall: {recall}"
    )
    # Calculate bias:
    count_labels = Counter(y_true)
    count_pred = Counter(y_argmax)
    bias = {}
    sum_counts = len(y_true)
    for y in count_labels.keys():
        if y in count_labels and y in count_pred:
            c_label = count_labels[int(y)] / sum_counts
            c_pred = count_pred[int(y)] / sum_counts
            b = c_pred - c_label
            bias[rot_categories[int(y)]] = b
        else:
            bias[rot_categories[int(y)]] = np.nan
    print(count_labels)
    print(count_pred)
    print(bias)
    with open(output_path / f"results_{suffix}.txt", "w") as f:
        f.write(f"Metrics AUC_OVR: {auc_ovr}\n")
        f.write(f"Metrics AUC_OVO: {auc_ovo}\n")
        f.write(f"Metrics Macro-Precision: {precision}")
        f.write(f"Metrics Macro-Recall: {recall}\n")
        f.write(
            f"Accuracy: {accuracy} \naccuracy_2: {accuracy_2}\naccuracy_3: {accuracy_3} \naccuracy_4: {accuracy_4}\naccuracy_5: {accuracy_5}\nprecision: {precision}\nrecall: {recall}\n"
        )
        f.write("Report:\n")
        f.write(f"{report}\n")
        f.write("Bias:\n")
        f.write(f"{bias}\n")
    weights = np.array([count_labels[r] for r in range(338)])
    weights = weights / np.sum(weights)
    unweighted_cm = confusion_matrix(
        y_true, y_argmax, normalize="all", labels=list(range(338))
    )
    create_rot_cm(
        unweighted_cm,
        rot_categories,
        mode=f"{suffix}_unweighted",
        output_path=output_path,
    )
    sample_weights = [weights[int(y)] for y in y_true]
    weighted_cm = confusion_matrix(
        y_true,
        y_argmax,
        normalize="all",
        sample_weight=sample_weights,
        labels=list(range(338)),
    )
    create_rot_cm(
        weighted_cm,
        rot_categories,
        mode=f"{suffix}_weighted",
        output_path=output_path,
    )


def extract_rotamer_encoding(pdb_code: str, monomer: ampal.Assembly) -> dict:
    """
    Extracts rotamer encoding from PDB structure.

    Parameters
    ----------
    pdb_code: str
        pdb code of the structure of interest
    monomer: ampal.Assembly
        Assembly of the pdb structure

    Returns
    -------
    dictionary: dict
        {pdb_code_monomer.id: all_rot}
    """
    _, flat_categories = get_rotamer_codec()
    res_rot_to_encoding = dict(zip(flat_categories, range(len(flat_categories))))
    all_rot = []
    for res in monomer:
        try:
            rot_key = f"{res.mol_code}_{''.join(map(str, res.tags['rotamers']))}"
            if rot_key in res_rot_to_encoding:
                all_rot.append(res_rot_to_encoding[rot_key])
            else:
                all_rot.append(np.nan)
        except TypeError:
            all_rot.append(np.nan)

    return {f"{pdb_code[:4]}{monomer.id}": all_rot}


def _tag_pdb_with_rot(pdb_code: str, pdb_path: Path) -> (dict, dict):
    """
    Tag pdb file with rotamer (note: the pdb file is not modified)

    Parameters
    ----------
    pdb_code: str
        PDB code for structure of interest
    pdb_path: Path
        Path to structure of interest

    Returns
    -------
    result_dict: dict
        {pdb_code_monomer.id: all_rot}
    assembly_dict: dict
        {pdb_code_monomer.id: ampal.Assembly}
    """
    if "_" in pdb_code:
        pdb_path = pdb_path / (pdb_code + ".pdb")
        if not pdb_path.exists():
            # Exit
            print(f"Could not find {pdb_path}")
            return (None, None)
    else:
        out_dir = pdb_path / pdb_code[1:3]
        pdb_path = out_dir / (pdb_code[:4] + ".pdb1.gz")
    if not pdb_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = out_dir / (pdb_code[:4] + ".pdb1")
        if not pdb_path.exists():
            pdb_path = _fetch_pdb(pdb_code, verbosity=1, output_folder=out_dir)
        assembly = ampal.load_pdb(pdb_path)
    else:
        if pdb_path.suffix == ".pdb":
            assembly = ampal.load_pdb(str(pdb_path))
        elif pdb_path.suffix == ".gz":
            with gzip.open(str(pdb_path), "rb") as inf:
                assembly = ampal.load_pdb(inf.read().decode(), path=False)
        else:
            raise ValueError(
                f"Expected filetype to be either .pdb or .gz but got {pdb_path.suffix} {pdb_path}"
            )
    if isinstance(assembly, ampal.AmpalContainer):
        assembly = assembly[0]
    if isinstance(assembly, ampal.Assembly):
        # For each monomer in the assembly:
        result_dict = {}
        for monomer in assembly:
            if isinstance(monomer, ampal.Polypeptide):
                monomer.tag_sidechain_dihedrals()
                curr_result_dict = extract_rotamer_encoding(pdb_code, monomer)
                result_dict.update(curr_result_dict)
    elif isinstance(assembly, ampal.Polypeptide):
        assembly.tag_sidechain_dihedrals()
        result_dict = extract_rotamer_encoding(pdb_code, assembly)
    # assembly_dict = {list(result_dict.keys())[0]: assembly}
    assembly_dict = {pdb_code[:4]: assembly}

    return result_dict, assembly_dict


def tag_pdb_with_rot(
    workers: int, path_to_pdb: Path, pdb_codes: np.ndarray
) -> (dict, dict):
    """
    Tag pdb file with rotamer (note: the pdb file is not modified).
    Uses multiprocessing

    Parameters
    ----------
    args: argparse
    path_to_pdb: Path
        Path to the pdb structures
    pdb_codes: np.ndarray
        List of pdb codes to tag

    Returns
    -------
    result_dict: dict
        {pdb_code_monomer.id: all_rot}
    assembly_dict: dict
        {pdb_code_monomer.id: ampal.Assembly}
    """
    results_dict = {}
    pdb_to_assemblies = {}
    # Loop through all the pdb structures and extract rotamer label
    with Pool(processes=workers) as p:
        results_dict_list = p.starmap(
            _tag_pdb_with_rot,
            zip(
                pdb_codes,
                repeat(path_to_pdb),
            ),
        )
        p.close()
    # Flatten dictionary:
    for curr_dict in results_dict_list:
        curr_res_dict, curr_assembly_dict = curr_dict
        if curr_res_dict is not None:
            results_dict.update(curr_res_dict)
            pdb_to_assemblies.update(curr_assembly_dict)

    return results_dict, pdb_to_assemblies
