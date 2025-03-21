import argparse
import typing as t
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from numpy import genfromtxt
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
from tqdm import tqdm

from design_utils.sampling_utils import apply_temp_to_probs, sample_with_multiprocessing
from design_utils.utils import (
    convert_dataset_map_for_srb,
    create_flat_dataset_map,
    extract_sequence_from_pred_matrix,
    load_batch,
)


@dataclass(frozen=True)
class ResidueTarget:
    pdb_id: str
    chain: str
    resnum: int
    target_aa: Optional[str] = None  # None = wild-type


def top_3_cat_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def get_residue_index_map(flat_dataset_map: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Sorts the flat dataset map by integer PDB residue index (column 3) and
    returns a mapping from (pdb_id, chain, pdb_resnum) to sequence index.

    Parameters
    ----------
    flat_dataset_map : np.ndarray
        Array of shape (N, 4+) with columns [pdb_id, chain_id, pdb_resnum, ...].

    Returns
    -------
    sorted_map : np.ndarray
        Dataset map sorted by integer residue number.
    residue_to_index : dict
        Mapping {(pdb_id, chain, pdb_resnum: int) -> index_in_sequence}
    """
    # Parse integer version of residue numbers
    resnums = flat_dataset_map[:, 2].astype(int)
    sorted_indices = np.argsort(resnums)
    sorted_map = flat_dataset_map[sorted_indices]

    residue_to_index = {
        (row[0], row[1], int(row[2])): i
        for i, row in enumerate(sorted_map)
    }

    return sorted_map, residue_to_index



def load_dataset_and_predict(
    model: Path,
    dataset_path: Path,
    path_to_output: Path,
    batch_size: int,
    sample_n: int,
    temperature: float = 1,
    workers: int = 1,
    residues_to_fix: Optional[t.List[ResidueTarget]] = None,
    residues_to_redesign: Optional[t.List[ResidueTarget]] = None,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Load discretized frame dataset (should be the same format as the trained models),
    creates a dataset map and predicts the frames using each of the models.

    Everything is then saved into a csv file.

    Parameters
    ----------
    model: Path
        Path to the trained model.
    dataset_path: Path
        Path to the dataset with frames.
    batch_size: int
        Number of frames to be looked predicted at once.
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
    # assume user passed HDF5 file TODO: Do this for PDB inputs as well

    # Import top3 accuracy:
    tf.keras.utils.get_custom_objects()["top_3_cat_acc"] = top_3_cat_acc
    path_to_datasetmap = path_to_output / f"{dataset_path.stem}.txt"

    # If dataset map exists, load it from path:
    if Path(path_to_datasetmap).exists():
        flat_dataset_map = genfromtxt(path_to_datasetmap, delimiter=",", dtype="str")
    else:
        # Create flat_map:
        flat_dataset_map, training_set_pdbs = create_flat_dataset_map(
            dataset_path,
        )
        # Save flat map to file:
        np.savetxt(path_to_datasetmap, flat_dataset_map, delimiter=",", fmt="%s")

    # Create a dictionary of position in the dataset map and position in the sequence
    flat_dataset_map, residue_to_index = get_residue_index_map(flat_dataset_map)
    # Calculate number of batches
    n_batches = ceil(len(flat_dataset_map) / batch_size)
    # Extract model name
    model_name = model.stem
    # Import Model:
    frame_model = tf.keras.models.load_model(model)
    # Create output file for model:
    path_to_prediction_probs = path_to_output / f"{dataset_path.stem}_model_{model_name}.csv"
    # Load batch:
    for index in tqdm(
        range(0, n_batches),
        desc=f"Processing batch of model {model_name}",
    ):
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
        # Save predictions to file:
        with open(path_to_prediction_probs, "a") as f:
            np.savetxt(f, y_pred_batch, delimiter=",")

    flat_dataset_map = np.array(flat_dataset_map)
    # Create output file for model:
    path_to_benchmark_map = path_to_output / f"{dataset_path.stem}_model_{model_name}.txt"
    # Output datasetmap compatible with sequence recovery benchmark:
    convert_dataset_map_for_srb(flat_dataset_map, model_name, path_to_benchmark_map)
    # Load prediction matrix
    prediction_matrix = genfromtxt(
        path_to_prediction_probs, delimiter=",", dtype=np.float16
    )
    # Apply temperature factor to prediction matrix:
    if temperature != 1:
        prediction_matrix = apply_temp_to_probs(prediction_matrix, t=temperature)

    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
    ) = extract_sequence_from_pred_matrix(
        flat_dataset_map,
        prediction_matrix,
    )
    pdb_codes = list(pdb_to_probability.keys())
    print(f"Ready to sample {sample_n} sequences for {len(pdb_codes)} proteins.")
    pdb_to_sampled = sample_with_multiprocessing(
        workers, pdb_codes, sample_n, pdb_to_probability
    )



def _check_residue_format(
    res_list: Optional[str], single_pdb_id: Optional[str] = None
) -> List[ResidueTarget]:
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
                raise ValueError(f"Invalid residue format: {item}")
            pdb_id, res = item.split(":")
        else:
            if not single_pdb_id:
                raise ValueError(f"Missing pdb_id for residue '{item}'")
            if not pattern_single.match(item):
                raise ValueError(f"Invalid residue format: {item}")
            pdb_id, res = single_pdb_id, item

        chain = res[0]
        i = 1
        while i < len(res) and res[i].isdigit():
            i += 1
        resnum = int(res[1:i])
        target_aa = res[i] if i < len(res) else None

        parsed.append(ResidueTarget(pdb_id, chain, resnum, target_aa))

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
        args.path_to_model,
        args.path_to_dataset,
        batch_size=args.batch_size,
        path_to_output=args.path_to_output,
        sample_n=args.sample_n,
        temperature=1,
        workers=args.workers,

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
    params.residues_to_fix = _check_residue_format(
        params.residues_to_fix, single_pdb_id=default_pdb_id
    )
    params.residues_to_redesign = _check_residue_format(
        params.residues_to_redesign, single_pdb_id=default_pdb_id
    )
    # Check for duplicates
    _check_duplicates(params.residues_to_fix, "residues_to_fix")
    _check_duplicates(params.residues_to_redesign, "residues_to_redesign")

    fix_keys = {(r.pdb_id, r.chain, r.resnum) for r in params.residues_to_fix}
    redesign_keys = {(r.pdb_id, r.chain, r.resnum) for r in params.residues_to_redesign}
    overlap = fix_keys & redesign_keys
    if overlap:
        raise ValueError(
            f"The same residues cannot be defined in both --residues_to_fix and --residues_to_redesign: {sorted(overlap)}. Please choose one."
        )

    main(params)
