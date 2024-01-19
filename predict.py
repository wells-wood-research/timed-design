import argparse
from math import ceil
from pathlib import Path

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
        model_out = path_to_output / ("{model_name}" + "_rot.csv" if predict_rotamers else f"{model_name}" + ".csv")
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
            save_outputs_to_file(y_true, y_pred, flat_dataset_map, i, model_name, path_to_output)
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


def main(args):
    # Sanitise paths
    args.path_to_dataset = Path(args.path_to_dataset)
    args.path_to_model = Path(args.path_to_model)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    args.path_to_output = Path(args.path_to_output)
    # check if output directory exists if not, ask the user if they want to create it
    if not args.path_to_output.exists():
        print(
            f"Output directory at {args.path_to_output} does not exist. Do you want to create it? (y/n)"
        )
        user_input = input()
        if user_input == "y":
            args.path_to_output.mkdir(parents=True, exist_ok=True)
        else:
            print("Exiting...")
            exit()

    if args.path_to_blacklist:
        args.path_to_blacklist = Path(args.path_to_blacklist)
        assert (
            args.path_to_blacklist.exists()
        ), f"Path to blacklist at {args.path_to_blacklist} does not exists."

    assert (
        args.path_to_model.exists()
    ), f"Path to model at {args.path_to_model} does not exists."
    assert (
        args.path_to_dataset.exists()
    ), f"Path to dataset at {args.path_to_dataset} does not exists."
    assert (
        args.batch_size > 0
    ), f"Batch size must be higher than 0 but got {args.batch_size}"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Number of batches of frames to predict at once (default: 12)",
    )
    parser.add_argument(
        "--path_to_dataset", type=str, help="Path to dataset file ending with .hdf5"
    )
    parser.add_argument(
        "--path_to_datasetmap",
        default="datasetmap.txt",
        type=str,
        help="Path to dataset map ending with .txt",
    )
    parser.add_argument(
        "--path_to_model", type=str, help="Path to model file ending with .h5"
    )
    parser.add_argument(
        "--path_to_blacklist",
        type=str,
        default=None,
        help="Path to csv file containing PDBs in the training set.",
    )
    parser.add_argument(
        "--path_to_output",
        type=str,
        default=".",
        help="Directory to save output files. Defaults to current working directory. If the directory does not exist, the user will be prompted to create it.",
    )
    parser.add_argument(
        "--output_analysis",
        action="store_true",
        help="Whether to output analysis graphs.",
    )
    parser.add_argument(
        "--predict_rotamers",
        action="store_true",
        help="Whether model outputs predictions for 338 rotamers (True) or 20 residues (False).",
    )
    parser.add_argument(
        "--is_structure_nmr",
        action="store_true",
        help="Whether the structure is NMR. NMR will have different states so TIMED will try to build a consensus",
    )
    params = parser.parse_args()
    main(params)
