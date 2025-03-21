import argparse
from math import ceil
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from numpy import genfromtxt
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tqdm import tqdm

from design_utils.utils import (
    convert_dataset_map_for_srb,
    create_flat_dataset_map,
    extract_sequence_from_pred_matrix,
    get_pdb_keys_to_filter,
    get_rotamer_codec,
    load_batch,
    save_consensus_probs,
    save_dict_to_fasta,
    save_outputs_to_file,
)

@dataclass(frozen=True)
class ResidueTarget:
    pdb_id: str
    chain: str
    resnum: int
    target_aa: Optional[str] = None  # None = wild-type

def top_3_cat_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def load_dataset_and_predict(
    models: list,
    dataset_path: Path,
    batch_size: int = 20,
    start_batch: int = 0,
    dataset_map_path: Path = "datasetmap.txt",
    blacklist: Path = None,
    predict_rotamers: bool = False,
    model_name_suffix: str = "",
    is_consensus: bool = False,
    path_to_output: Path = Path.cwd(),
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load discretized frame dataset (should be the same format as the trained models),
    creates a dataset map and predicts the frames using each of the models.

    Everything is then saved into a csv file.

    Parameters
    ----------
    models: t.List[StrOrPath]
        List of paths to the models to be used for the ensemble
    dataset_path: Path
        Path to the dataset with frames.
    batch_size: int
        Number of frames to be looked predicted at once.
    start_batch:
        Which batch to start from. In case the code crashes you can check which
        was the last batch used and restart from there. Make sure you remove the
        other models from the paths to be used.
    dataset_map_path: Path
        Path to the dataset map
    blacklist: Path
        Path to blacklist of structures to be filtered out (ie. not predicted)
    predict_rotamers: Bool
        Whether to predict 338 classes of rotamers or just the 20 amino acids
    model_name_suffix: str
        Suffix to be added to predictions which indicates model name
    is_consensus: Bool
        Whether the structure is NMR and the prediction should be a consensus of all the states
    path_to_output: Path
        Path to output directory. Defaults to current working directory.

    Returns
    -------
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    pdb_to_sequence: dict
        Dictionary {pdb_code: predicted_sequence}
    pdb_to_probability: dict
        Dictionary {pdb_code: probability}
    pdb_to_real_sequence: dict
        Dictionary {pdb_code: sequence}
    pdb_to_consensus: dict
        Dictionary {pdb_code: consensus_sequence}
    pdb_to_consensus_prob: dict
        Dictionary {pdb_code: consensus_probability}
    """
    # Import top3 accuracy:
    tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc

    n_classes = 338 if predict_rotamers else 20
    print(f"Running model on {n_classes} classes. Rotamer Mode is {predict_rotamers}")
    # Get list of banned pdbs from the benchmark:
    if blacklist:
        filter_pdb_list = get_pdb_keys_to_filter(blacklist)
    else:
        filter_pdb_list = []
    # If dataset map exists, load it from path:
    if Path(dataset_map_path).exists():
        flat_dataset_map = genfromtxt(dataset_map_path, delimiter=",", dtype="str")
    else:
        # Create flat_map:
        flat_dataset_map, training_set_pdbs = create_flat_dataset_map(
            dataset_path, filter_pdb_list
        )
    old_datasetmap = True if len(flat_dataset_map[0]) == 4 else False

    if predict_rotamers:
        codec, flat_categories = get_rotamer_codec()
    else:
        codec, flat_categories = None, None
    # Calculate number of batches
    n_batches = ceil(len(flat_dataset_map) / batch_size)
    # For each model:
    for i, m in enumerate(models):
        # Extract model names:
        if isinstance(m, Path):
            model_name = m.stem + model_name_suffix
        else:
            model_name = str(m) + model_name_suffix
        # Import Model:
        frame_model = tf.keras.models.load_model(Path(m))
        # Create output file for model:
        model_out = path_to_output / (
            "{model_name}" + "_rot.csv"
            if predict_rotamers
            else f"{model_name}" + ".csv"
        )
        # Load batch:
        for index in tqdm(
            range(start_batch, n_batches),
            desc=f"Processing batch of model {model_name}",
        ):
            # Initialize array for predictions:
            y_true = []
            # Initialize dictionary with {model_number : [predictions]}
            y_pred = {k: [] for k in range(len(models))}
            # Extract current batch map:
            current_batch_map = flat_dataset_map[
                index * batch_size : (index + 1) * batch_size
            ]
            X_batch, y_true_batch = load_batch(
                dataset_path,
                current_batch_map,
            )
            # Make Predictions
            y_pred_batch = frame_model.predict(X_batch)
            if predict_rotamers:
                # Output model predictions:
                with open(model_out, "a") as f:
                    np.savetxt(f, y_pred_batch, delimiter=",")
                current_batch = np.argmax(y_pred_batch, axis=1)
                y_pred_batch = np.array([codec[c] for c in current_batch])
                del current_batch
            # Add predictions labels to dictionary:
            y_pred[i].extend(y_pred_batch)
            # Save current labels:
            y_true.extend(y_true_batch)
            # Save to output file:
            save_outputs_to_file(
                y_true, y_pred, flat_dataset_map, i, model_name, path_to_output
            )
            # Reset to avoid memory errors
            del y_true
            del y_pred
        flat_dataset_map = np.array(flat_dataset_map)
        # Output datasetmap compatible with sequence recovery benchmark:
        convert_dataset_map_for_srb(flat_dataset_map, model_name, path_to_output)
        # Load prediction matrix
        prediction_matrix = genfromtxt(model_out, delimiter=",", dtype=np.float16)
        # Save as Fasta file:
        (
            pdb_to_sequence,
            pdb_to_probability,
            pdb_to_real_sequence,
            pdb_to_consensus,
            pdb_to_consensus_prob,
        ) = extract_sequence_from_pred_matrix(
            flat_dataset_map,
            prediction_matrix,
            rotamers_categories=flat_categories if predict_rotamers else None,
            old_datasetmap=old_datasetmap,
            is_consensus=is_consensus,
        )
        save_dict_to_fasta(pdb_to_sequence, model_name, path_to_output)
        save_dict_to_fasta(pdb_to_real_sequence, "dataset", path_to_output)
        if pdb_to_consensus:
            save_dict_to_fasta(
                pdb_to_consensus,
                model_name + "_consensus",
            )
            save_consensus_probs(pdb_to_consensus_prob, model_name, path_to_output)

    return (
        flat_dataset_map,
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    )


def _check_residue_format(
    res_list: Optional[str], single_pdb_id: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Validates and parses a list of residue identifiers for redesign or fixation.

    Each residue identifier must be in one of the following formats:
        - <pdb_id>:<chain><resnum> (e.g., 1XYZ:A12)
        - <pdb_id>:<chain><resnum><AA> (e.g., 1XYZ:A12P)
        - <chain><resnum> (e.g., A12), if single_pdb_id is provided
        - <chain><resnum><AA> (e.g., A12P), if single_pdb_id is provided

    Parameters
    ----------
    res_list : str or None
        Comma-separated string of residue identifiers to fix or redesign.
        Examples: "A12,A35P", "1XYZ:A12,2ABC:B20R"

    single_pdb_id : str or None
        Default PDB ID if none is provided in the identifiers.

    Returns
    -------
    List[Tuple[str, str]]
        List of (pdb_id, residue_str), where residue_str is like "A12" or "A12P".

    Raises
    ------
    ValueError
        If residue identifiers do not match expected formats, or if PDB ID is
        missing in multi-PDB context.
    """
    if res_list is None:
        return []

    pattern_with_pdb = re.compile(r"^[a-zA-Z0-9]+:[A-Za-z]\d+[A-Za-z]?$")
    pattern_single = re.compile(r"^[A-Za-z]\d+[A-Za-z]?$")

    parsed = []
    for item in res_list.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            if not pattern_with_pdb.match(item):
                raise ValueError(
                    f"Invalid residue format: '{item}'. Expected format: <pdb_id>:<chain><resnum>[<AA>], e.g., 1XYZ:A12 or 1XYZ:A12P"
                )
            pdb_id, res = item.split(":")
        else:
            if not single_pdb_id:
                raise ValueError(f"Residue '{item}' missing pdb_id in multi-PDB mode.")
            if not pattern_single.match(item):
                raise ValueError(
                    f"Invalid residue format: '{item}'. Expected format: <chain><resnum>[<AA>], e.g., A12 or A12P"
                )
            pdb_id, res = single_pdb_id, item

        parsed.append((pdb_id, res))

    return parsed

def _check_duplicates(residue_list: List[ResidueTarget], label: str):
    seen = set()
    for r in residue_list:
        key = (r.pdb_id, r.chain, r.resnum)
        if key in seen:
            raise ValueError(f"Duplicate entry in {label}: {key}")
        seen.add(key)

def main(args):
    (
        flat_dataset_map,
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    ) = load_dataset_and_predict(
        [args.path_to_model],
        args.path_to_dataset,
        batch_size=args.batch_size,
        start_batch=0,
        blacklist=args.path_to_blacklist,
        dataset_map_path=args.path_to_datasetmap,
        predict_rotamers=args.predict_rotamers,
        is_consensus=args.is_structure_nmr,
        path_to_output=args.path_to_output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with TIMED")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Number of frames to predict at once (default: 12)",
    )
    parser.add_argument(
        "--path_to_dataset", type=Path, help="Path to dataset file ending with .hdf5"
    )
    parser.add_argument(
        "--path_to_model", type=Path, help="Path to model file ending with .h5"
    )
    parser.add_argument(
        "--path_to_output",
        type=Path,
        default=".",
        help="Directory to save output files. Defaults to current working directory. If the directory does not exist, the user will be prompted to create it.",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=100,
        help="Number of samples to be drawn from the distribution.",
    )
    parser.add_argument(
        "--residues_to_fix",
        type=str,
        default=None,
        help=(
            "Comma-separated list of residues to fix. Format: <chain><resnum> for wild-type (e.g., A12), or <chain><resnum><AA> to fix to specific amino acid (e.g., A12P)."
        ),
    )

    parser.add_argument(
        "--residues_to_redesign",
        type=str,
        default=None,
        help="Comma-separated residues to redesign. Format: <pdb_id>:<res><chain> (e.g., 2ABC:42B). If only one PDB, use <res><chain>.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    params = parser.parse_args()

    # Check paths:
    if not params.path_to_output.exists():
        params.path_to_output.mkdir(parents=True, exist_ok=True)
    assert (
        params.path_to_model.exists()
    ), f"Path to model at {params.path_to_model} does not exists."
    assert (
        params.path_to_dataset.exists()
    ), f"Path to dataset at {params.path_to_dataset} does not exists."
    assert (
        params.batch_size > 0
    ), f"Batch size must be higher than 0 but got {params.batch_size}"
    # Check residues to fix and redesign:
    if params.residues_to_fix and params.residues_to_redesign:
        raise ValueError(
            "Cannot fix and redesign residues at the same time. Please choose one."
        )
    # Check residue format
    default_pdb_id = "default"
    fix_targets_raw = _check_residue_format(params.residues_to_fix, single_pdb_id=default_pdb_id)
    redesign_targets_raw = _check_residue_format(params.residues_to_redesign, single_pdb_id=default_pdb_id)
    # Check for duplicates
    _check_duplicates(fix_targets_raw, "residues_to_fix")
    _check_duplicates(redesign_targets_raw, "residues_to_redesign")
    # Check for cross-conflicts
    fix_keys = {(r.pdb_id, r.chain, r.resnum) for r in fix_targets_raw}
    redesign_keys = {(r.pdb_id, r.chain, r.resnum) for r in redesign_targets_raw}

    overlap = fix_keys & redesign_keys
    if overlap:
        raise ValueError(f"Residues defined in both --residues_to_fix and --residues_to_redesign: {sorted(overlap)}")
    main(params)
