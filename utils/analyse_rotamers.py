import argparse
import gzip
from collections import Counter
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import typing as t
from ampal.amino_acids import standard_amino_acids
from isambard.modelling import scwrl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

import ampal
from aposteriori.data_prep.create_frame_data_set import _fetch_pdb
from utils import extract_sequence_from_pred_matrix, get_rotamer_codec


def save_assembly_to_path(structure: ampal.Assembly, output_dir: Path, name: str):
    output_path = output_dir / (name + ".pdb")
    with open(output_path, "w") as f:
        f.write(structure.pdb)


def pack_sidechains(structure: ampal.Assembly, sequence: str) -> ampal.Assembly:
    return scwrl.pack_side_chains_scwrl(
        assembly=structure, sequences=sequence, rigid_rotamer_model=False
    )


def analyse_with_scwrl(
    pdb_to_seq: dict, pdb_to_assembly: dict, output_path: Path, suffix: str
):
    for pdb in tqdm(pdb_to_seq.keys(), desc="Packing sequence in PDB with SCWRL"):
        pdb_outpath = output_path / (pdb + "_" + suffix + ".pdb")
        if pdb_outpath.exists():
            print(f"PDB {pdb} at {pdb_outpath} already exists.")
        elif pdb in pdb_to_assembly.keys():
            try:
                if len(pdb_to_assembly[pdb].backbone) > 1:
                    # pdb_to_assembly[pdb] = ampal.Assembly(pdb_to_assembly[pdb][0])
                    pdb_to_seq[pdb] = [pdb_to_seq[pdb]] * len(pdb_to_assembly[pdb])
                else:
                    pdb_to_seq[pdb] = [pdb_to_seq[pdb]]
                try:
                    scwrl_structure = pack_sidechains(
                        pdb_to_assembly[pdb], pdb_to_seq[pdb]
                    )
                    save_assembly_to_path(
                        structure=scwrl_structure,
                        output_dir=output_path,
                        name=pdb + suffix,
                    )
                except ValueError as e:
                    print(f"Attempted packing on structure {pdb}, but got {e}")
            except ValueError as e:
                print(f"Attempted selecting backbone on structure {pdb}, but got {e}")
            except KeyError as e:
                print(f"Attempted selecting backbone on structure {pdb}, but got {e}")
            except ChildProcessError as e:
                print(
                    f"Attempted selecting backbone on structure {pdb}, but SCWRL failed: {e}"
                )
        else:
            print(f"Error with structure {pdb}. Assembly not found.")


def plot_cm(
    cm: np.ndarray,
    y_labels: t.List[str],
    x_labels: t.List[str],
    title: str,
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
    fig.savefig(f"{title.replace(' ', '_')}.png")
    # Save Confusion:
    plt.close()


def create_rot_cm(cm: np.ndarray, rot_categories: t.List[str], mode: str):
    """

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
        )
        if len(small_cm) > 1:  # Avoids bug with glycine and alanine
            plot_cm(
                small_cm,
                y_labels=curr_rot_cat,
                x_labels=curr_rot_cat,
                title=f"{mode} {res} vs {res} rot",
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
        )


def calulate_metrics(
    pdb_to_probability: dict, pdb_to_rotamer: dict, rot_categories: t.List[str], suffix: str
):
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
            y_pred += pdb_to_probability[pdb]
            y_true += pdb_to_rotamer[pdb]
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
    with open(f"results_{suffix}.txt", "w") as f:
        f.write(f"Metrics AUC_OVR: {auc_ovr}\n")
        f.write(f"Metrics AUC_OVO: {auc_ovo}\n")
        f.write(f"Metrics Macro-Precision: {precision} Macro-Recall: {recall}\n")
        f.write(
            f"Accuracy: {accuracy}, accuracy_2: {accuracy_2}, accuracy_3: {accuracy_3}, accuracy_4: {accuracy_4}, accuracy_5: {accuracy_5}, precision: {precision}, recall: {recall}\n"
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
    create_rot_cm(unweighted_cm, rot_categories, mode=f"{suffix}_unweighted")
    sample_weights = [weights[int(y)] for y in y_true]
    weighted_cm = confusion_matrix(
        y_true,
        y_argmax,
        normalize="all",
        sample_weight=sample_weights,
        labels=list(range(338)),
    )
    create_rot_cm(weighted_cm, rot_categories, mode=f"{suffix}_weighted")


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
            raise ValueError(f"Expected filetype to be either .pdb or .gz but got {pdb_path.suffix} {pdb_path}")
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
    assembly_dict = {list(result_dict.keys())[0]: assembly}

    return result_dict, assembly_dict


def tag_pdb_with_rot(workers: int, path_to_pdb: Path, pdb_codes: np.ndarray) -> (dict, dict):
    """

    Parameters
    ----------
    args
    pdb_codes

    Returns
    -------

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


def main(args):
    args.input_path = Path(args.input_path)
    model_name = args.input_path.stem
    args.output_path = Path(f"{args.output_path}_{model_name}")
    # Create output folder if it does not exist:
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    args.path_to_pdb = Path(args.path_to_pdb)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Datasetmap file {args.path_to_datasetmap} does not exist"
    assert args.path_to_pdb.exists(), f"PDB folder {args.path_to_pdb} does not exist"
    # Load datasetmap
    datasetmap = np.genfromtxt(f"{args.path_to_datasetmap}", delimiter=",", dtype=str)
    # Extract PDB codes
    pdb_codes = np.unique(datasetmap[:, 0])
    results_dict, pdb_to_assemblies = tag_pdb_with_rot(args.workers, args.path_to_pdb, pdb_codes)
    # Load prediction matrix of model of interest:
    prediction_matrix = np.genfromtxt(args.input_path, delimiter=",", dtype=np.float16)
    # Get rotamer categories:
    _, flat_categories = get_rotamer_codec()
    # Get dictionary for 3 letter -> 1 letter conversion:
    res_to_r = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
    # Create flat categories of 1 letter amino acid for each of the 338 rotamers:
    rotamers_categories = [res_to_r[res.split("_")[0]] for res in flat_categories]
    # Extract dictionaries with sequences:
    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    ) = extract_sequence_from_pred_matrix(
        datasetmap, prediction_matrix, rotamers_categories=rotamers_categories
    )
    # Calculate Metrics:
    # - Analysis 1: TIMED_rotamer vs real rotamers from crystal structure
    calulate_metrics(pdb_to_probability, results_dict, flat_categories, suffix=f"{model_name}_vs_original")
    # TODO: Save metrics
    # Analyse rotamers with SCWRL (requires SCWRL install)
    # First the sequence is packed with SCWRL and saved to PDB,
    # Then, the same metrics as before are calculated and saved
    # - Analysis 2: TIMED_rotamer vs TIMED sequence put through SCWRL
    # analyse_with_scwrl(
    #     pdb_to_sequence, pdb_to_assemblies, args.output_path, suffix=model_name
    # )
    # model_pdb_codes = np.core.defchararray.add(pdb_codes, model_name)
    # model_results_dict, _ = tag_pdb_with_rot(args.workers,  args.output_path, model_pdb_codes)
    # calulate_metrics(pdb_to_probability, model_results_dict, flat_categories, suffix=f"{model_name}_vs_pred+scwrl")
    # TODO: Save metrics
    # - Analysis 3: TIMED_rotamer vs Real sequence from crystal put through SCWRL
    # analyse_with_scwrl(
    #     pdb_to_real_sequence, pdb_to_assemblies, args.output_path, suffix="scwrl"
    # )
    # scwrl_pdb_codes = np.core.defchararray.add(pdb_codes, ("_" + "scwrl"))
    # scwrl_results_dict, _ = tag_pdb_with_rot(args.workers, args.path_to_pdb, scwrl_pdb_codes)
    # calulate_metrics(pdb_to_probability, scwrl_results_dict, flat_categories, suffix=f"{model_name}_vs_ori+scwrl")
    # # TODO: Save metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to model .csv file")
    parser.add_argument(
        "--output_path", default="output", type=str, help="Path to save analysis"
    )
    parser.add_argument(
        "--path_to_pdb",
        type=str,
        help="Path to biounit pdb dataset. Needs to be in format pdb/{2nd and 3rd char}/{pdb}.pdb1.gz",
    )
    parser.add_argument(
        "--path_to_datasetmap",
        default="datasetmap.txt",
        type=str,
        help="Path to dataset map ending with .txt",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    params = parser.parse_args()
    main(params)