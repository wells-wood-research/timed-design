import sys
import typing as t
import warnings
from itertools import product
from pathlib import Path

import h5py
import numpy as np
from ampal.amino_acids import side_chain_dihedrals, standard_amino_acids
from numpy import genfromtxt

from aposteriori.config import MAKE_FRAME_DATASET_VER, UNCOMMON_RESIDUE_DICT
from aposteriori.data_prep.create_frame_data_set import DatasetMetadata


def lookup_blosum62(res_true: str, res_prediction: str) -> int:
    """Returns score from the matrix.
    Parameters
    ----------
    res_true: str
        First residue code.
    res_prediction: str
        Second residue code.
    Returns
    --------
    Score from the matrix."""

    if (res_true, res_prediction) in blosum62.keys():
        return blosum62[res_true, res_prediction]
    else:
        return blosum62[res_prediction, res_true]


def load_datasetmap(path_to_datasetmap: Path, is_old: bool = False) -> np.ndarray:
    """
    Load dataset map. Supports old datasetmap pre-benchmark.

    Parameters
    ----------
    path_to_datasetmap: Path
        Path to the datasetmap to be loaded
    is_old: bool
        Whether the datasetmap is old. Note: this allows for backwards compatibility.

    Returns
    -------
    dataset_map: np.ndarray
        2D array of datasetmap
    """
    assert (
        path_to_datasetmap.suffix == ".txt"
    ), f"Expected Path {path_to_datasetmap} to be a .txt file but got {path_to_datasetmap.suffix}."
    if is_old:
        dataset_map = np.genfromtxt(
            path_to_datasetmap,
            delimiter=",",
            dtype=str,
        )
    else:
        dataset_map = np.genfromtxt(
            path_to_datasetmap,
            delimiter=" ",
            dtype=str,
            skip_header=3,
        )
    # If list only contains 1 pdb, it fails to create a list of list [pdb_code, count]
    if len(dataset_map) == 2:
        dataset_map = [dataset_map]

    return dataset_map


def extract_metadata_from_dataset(frame_dataset: Path) -> DatasetMetadata:
    """
    Retrieves the metadata of the dataset and does a sanity check of the version.
    If the dataset version is not compatible with aposteriori, the training process will stop.

    Parameters
    ----------
    frame_dataset: Path
        Path to the .h5 dataset with the following structure.
        ??????[pdb_code] Contains a number of subgroups, one for each chain.
          ??????[chain_id] Contains a number of subgroups, one for each residue.
            ??????[residue_id] voxels_per_side^3 array of ints, representing element number.
              ??????.attrs['label'] Three-letter code for the residue.
              ??????.attrs['encoded_residue'] One-hot encoding of the residue.
        ??????.attrs['make_frame_dataset_ver']: str - Version used to produce the dataset.
        ??????.attrs['frame_dims']: t.Tuple[int, int, int, int] - Dimentsions of the frame.
        ??????.attrs['atom_encoder']: t.List[str] - Lables used for the encoding (eg, ["C", "N", "O"]).
        ??????.attrs['encode_cb']: bool - Whether a Cb atom was added at the avg position of (-0.741287356, -0.53937931, -1.224287356).
        ??????.attrs['atom_filter_fn']: str - Function used to filter the atoms in the frame.
        ??????.attrs['residue_encoder']: t.List[str] - Ordered list of residues corresponding to the encoding used.
        ??????.attrs['frame_edge_length']: float - Length of the frame in Angstroms (A)
        ??????.attrs['voxels_as_gaussian']: bool - Whether the voxels are encoded as a floating point of a gaussian (True) or boolean (False)


    Returns
    -------
    dataset_metadata: DatasetMetadata of the dataset with the following parameters:
        make_frame_dataset_ver: str
        frame_dims: t.Tuple[int, int, int, int]
        atom_encoder: t.List[str]
        encode_cb: bool
        atom_filter_fn: str
        residue_encoder: t.List[str]
        frame_edge_length: float
        voxels_as_gaussian: bool

    """
    with h5py.File(frame_dataset, "r") as dataset_file:
        meta_dict = dict(dataset_file.attrs.items())
        dataset_metadata = DatasetMetadata.import_metadata_dict(meta_dict)

    # Extract version metadata:
    dataset_ver_num = dataset_metadata.make_frame_dataset_ver.split(".")[0]
    aposteriori_ver_num = MAKE_FRAME_DATASET_VER.split(".")[0]
    # If the versions are compatible, return metadata else stop:
    if dataset_ver_num != aposteriori_ver_num:
        sys.exit(
            f"Dataset version is {dataset_metadata.make_frame_dataset_ver} and is incompatible "
            f"with Aposteriori version {MAKE_FRAME_DATASET_VER}."
            f"Try re-creating the dataset with the current version of Aposteriori."
        )
    return dataset_metadata


def get_pdb_keys_to_filter(
    pdb_key_path: Path, file_extension: str = ".txt"
) -> t.List[str]:
    """
    Obtains list of PDB keys from benchmark file. This is to ensure no leakage
    of training samples is seen in the benchmark.

    Parameters
    ----------
    pdb_key_path: Path
        Path to files with pdb keys.
    file_extension: str
        Extension of file. Defaults to ".txt"

    Returns
    -------
    pdb_keys_list: t.List[str]
        List of pdb keys to be removed from training set.
    """
    pdb_key_files = list(pdb_key_path.glob(f"**/*{file_extension}"))
    assert len(pdb_key_files) >= 1, "Expected at least 1 pdb key file."

    pdb_keys_list = []
    # For each file:
    for pdb_list_file in pdb_key_files:
        curr_keys_list = genfromtxt(pdb_list_file, dtype=str)
        # filter chain (we want to delete the whole structure, regardless of chain:
        for pdb in curr_keys_list:
            # Add to list:
            pdb_keys_list.append(pdb[:4])

    return pdb_keys_list


def create_flat_dataset_map(
    frame_dataset: Path,
    filter_list: t.List[str] = [],
    remove_blacklist_silently: bool = False,
) -> (t.List[t.Tuple[str, int, str, str]], t.Set[str]):
    """
    Flattens the structure of the h5 dataset for batching and balancing
    purposes.

    Parameters
    ----------
    frame_dataset: Path
        Path to the .h5 dataset with the following structure.
        ??????[pdb_code] Contains a number of subgroups, one for each chain.
          ??????[chain_id] Contains a number of subgroups, one for each residue.
            ??????[residue_id] voxels_per_side^3 array of ints, representing element number.
              ??????.attrs['label'] Three-letter code for the residue.
              ??????.attrs['encoded_residue'] One-hot encoding of the residue.
        ??????.attrs['make_frame_dataset_ver']: str - Version used to produce the dataset.
        ??????.attrs['frame_dims']: t.Tuple[int, int, int, int] - Dimentsions of the frame.
        ??????.attrs['atom_encoder']: t.List[str] - Lables used for the encoding (eg, ["C", "N", "O"]).
        ??????.attrs['encode_cb']: bool - Whether a Cb atom was added at the avg position of (-0.741287356, -0.53937931, -1.224287356).
        ??????.attrs['atom_filter_fn']: str - Function used to filter the atoms in the frame.
        ??????.attrs['residue_encoder']: t.List[str] - Ordered list of residues corresponding to the encoding used.
        ??????.attrs['frame_edge_length']: float - Length of the frame in Angstroms (A)
    filter_list: t.List[str]
        List of banned PDBs. These are automatically removed from the train/validation set.
    remove_blacklist_silently: bool
        Whether to remove the pdb codes in the blacklist with a warning (True), or raise ValueError (False and default)
    Returns
    -------
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    training_set_pdbs: set
        Set of all the pdb codes in the training/validation set.
    """
    standard_residues = list(standard_amino_acids.values())
    # Training set pdbs:
    training_set_pdbs = set()

    with h5py.File(frame_dataset, "r") as dataset_file:
        flat_dataset_map = []
        # Create flattened dataset structure:
        for pdb_code in dataset_file:
            # Check first 4 letters of PBD code in blacklist:
            if pdb_code[:4] not in filter_list:
                for chain_id in dataset_file[pdb_code].keys():
                    # Sort by residue int rather than str
                    residue_n = np.array(
                        list(dataset_file[pdb_code][chain_id].keys()), dtype=np.int
                    )
                    residue_n.sort()
                    residue_n = np.array(residue_n, dtype=str)
                    for residue_id in residue_n:
                        # Extract residue info:
                        residue_label = dataset_file[pdb_code][chain_id][
                            str(residue_id)
                        ].attrs["label"]

                        if residue_label in standard_residues:
                            pass
                        # If uncommon, attempt conversion of label
                        elif residue_label in UNCOMMON_RESIDUE_DICT.keys():
                            warnings.warn(f"{residue_label} is not a standard residue.")
                            # Convert residue to common residue
                            residue_label = UNCOMMON_RESIDUE_DICT[residue_label]
                            warnings.warn(f"Residue converted to {residue_label}.")
                        else:
                            assert (
                                residue_label in standard_residues
                            ), f"Expected natural amino acid, but got {residue_label}."

                        flat_dataset_map.append(
                            (pdb_code, chain_id, residue_id, residue_label)
                        )
                        training_set_pdbs.add(pdb_code)
            else:
                if remove_blacklist_silently:
                    warnings.warn(
                        f"PDB code {pdb_code} was found in benchmark dataset. It was automatically removed."
                    )
                else:
                    raise ValueError(
                        f"PDB code {pdb_code} was found in benchmark dataset. "
                        f"Turn on remove_blacklist_silently=True if you want to"
                        f" ignore these structures for training."
                    )

    return flat_dataset_map, training_set_pdbs


def get_rotamer_codec() -> dict:
    """
    Creates a codec for tagging residues rotamers.
    Returns
    -------
    res_rot_to_encoding: dict
        Rotamer residues encoding of the format {1_letter_res : {rotamer_tuple: encoding}}
    """
    res_rot_to_encoding = {}
    flat_categories = []
    rot_to_20res = {}
    all_count = 338
    r_count = 0  # Number of rotamers processed so far
    for i, (a, res) in enumerate(standard_amino_acids.items()):
        if res in side_chain_dihedrals:
            n_rot = len(side_chain_dihedrals[res])
            all_rotamers = list(product([1, 2, 3], repeat=n_rot))
            encoding = np.arange(r_count, r_count + len(all_rotamers))
            onehot_encoding = np.zeros((len(all_rotamers), all_count))
            # Encodings are sorted so we can do encoding encoding
            onehot_encoding[np.arange(0, len(encoding)), encoding] = 1
            rot_to_encoding = dict(zip(all_rotamers, onehot_encoding))
            res_rot_to_encoding[res] = rot_to_encoding
            all_rotamers = np.array(all_rotamers, dtype=str)
            for r, rota in enumerate(all_rotamers):
                flat_categories.append(f"{res}_{''.join(rota)}")
                rot_to_20res[r_count + r] = np.array([0] * 20)
                rot_to_20res[r_count + r][i] = 1
            r_count += len(all_rotamers)
        # No rotamers available:
        else:
            n_rot = 1
            onehot_encoding = np.array([0] * all_count)
            onehot_encoding[r_count] = 1
            rot_to_encoding = {(0,): onehot_encoding}
            res_rot_to_encoding[res] = rot_to_encoding
            flat_categories.append(f"{res}_0")
            rot_to_20res[r_count] = np.array([0] * 20)
            rot_to_20res[r_count][i] = 1
            r_count += n_rot

    return rot_to_20res, flat_categories


def load_batch(
    dataset_path: Path,
    data_point_batch: t.List[t.Tuple],
) -> (np.ndarray, np.ndarray):
    """
    Load batch from a dataset map.

    Parameters
    ----------
    dataset_path: Path
        Path to the dataset
    data_point_batch: t.List[t.Tuple]
        Flat dataset map of current batch

    Returns
    -------
    X: np.ndarray
        5D frames with (batch_size, n, n, n, n_encoding) shape
    y: np.ndarray
        Array of shape (batch_size, 20) containing labels of frames
        or (batch_size, 338) if predict_rotamers=True

    """
    # Calcualte catch size
    batch_size = len(data_point_batch)
    remove_idx = []
    # Open hdf5:
    with h5py.File(str(dataset_path), "r") as dataset:
        dims = dataset.attrs["frame_dims"]
        voxels_as_gaussian = dataset.attrs["voxels_as_gaussian"]
        # Initialize X and y:
        if voxels_as_gaussian:
            X = np.zeros((batch_size, *dims), dtype=float)
        else:
            X = np.zeros((batch_size, *dims), dtype=bool)
        y = np.zeros((batch_size, 20), dtype=float)
        # Extract frame from batch:
        for i, (pdb_code, chain_id, residue_id, _) in enumerate(data_point_batch):
            # Extract frame:
            residue_frame = np.asarray(dataset[pdb_code][chain_id][residue_id][()])
            X[i] = residue_frame
            # Extract residue label:
            y[i] = dataset[pdb_code][chain_id][residue_id].attrs["encoded_residue"]
    return X, y


def convert_dataset_map_for_srb(flat_dataset_map: list, model_name: str):
    """

    Parameters
    ----------
    flat_dataset_map
    model_name

    Returns
    -------

    """
    count_dict = {}
    for i, (pdb, chain, res_idx, _) in enumerate(flat_dataset_map):
        if "_0" in pdb:
            pdb = pdb.split("_0")[0]
        if len(pdb) == 4:
            pdb += chain
        if pdb not in count_dict:
            count_dict[pdb] = 0

        count_dict[pdb] += 1

    with open(f"{model_name}.txt", "w") as d:
        d.write("ignore_uncommon False\ninclude_pdbs\n##########\n")
        for pdb, count in count_dict.items():
            d.write(f"{pdb} {count}\n")


def save_consensus_probs(pdb_to_consensus_prob: dict, model_name: str):
    """
    Saves consensus sequence into PDBench-compatible format.

    Parameters
    ----------
    pdb_to_consensus_prob: dict

    model_name: dict

    """
    with open(f"{model_name}_consensus.txt", "w") as d, open(
        f"{model_name}_consensus.csv", "a"
    ) as p:
        d.write("ignore_uncommon False\ninclude_pdbs\n##########\n")
        for pdb, predictions in pdb_to_consensus_prob.items():
            d.write(f"{pdb} {len(predictions)}\n")
            np.savetxt(p, predictions, delimiter=",")


def save_dict_to_fasta(pdb_to_sequence: dict, model_name: str):
    """
    Saves a dictionary of protein sequences to a fasta file.

    Parameters
    ----------
    pdb_to_sequence: dict
        Dictionary {pdb_code: predicted_sequence}
    model_name: str
        Name of the model.
    """
    with open(f"{model_name}.fasta", "w") as f:
        for pdb, seq in pdb_to_sequence.items():
            f.write(f">{pdb}\n{seq}\n")


def extract_sequence_from_pred_matrix(
    flat_dataset_map: t.List[t.Tuple],
    prediction_matrix: np.ndarray,
    rotamers_categories: t.List[str],
    old_datasetmap: bool = False,
) -> (dict, dict, dict, dict, dict):
    """
    Extract sequence from prediction matrix and create pdb_to_sequence and
    pdb_to_probability dictionaries

    Parameters
    ----------
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    prediction_matrix: np.ndarray
        Prediction matrix for each of the sequence

    Returns
    -------
    pdb_to_sequence: dict
        Dictionary {pdb_code: predicted_sequence}
    pdb_to_sequence: dict
        Dictionary {pdb_code: sequence}
    pdb_to_probability: dict
        Dictionary {pdb_code: probability}
    """
    pdb_to_sequence = {}
    pdb_to_probability = {}
    pdb_to_real_sequence = {}
    pdb_to_consensus = {}
    pdb_to_consensus_prob = {}
    # Whether the dataset contains multiple states of NMR or not
    is_consensus = False
    res_to_r_dic = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
    if rotamers_categories:
        if len(rotamers_categories[0]) == 1:
            res_dic = rotamers_categories
        else:
            res_dic = [res_to_r_dic[res.split("_")[0]] for res in rotamers_categories]
    else:
        res_dic = list(standard_amino_acids.keys())
    # Extract max idx for prediction matrix:
    max_idx = np.argmax(prediction_matrix, axis=1)
    # Loop through dataset map to create dictionaries:
    previous_count = 0
    old_datasetmap = True if len(flat_dataset_map[0]) == 4 else False
    for i in range(len(flat_dataset_map)):
        # Add support for different dataset maps:
        if old_datasetmap:
            pdb, chain, _, res = flat_dataset_map[i]
            count = 1
        else:
            pdb, count = flat_dataset_map[i]
            count = int(count)
            # chain = ""
        if "_" in pdb:
            pdbchain = pdb
            is_consensus = True
        else:
            if len(pdb) == 5:
                pdbchain = pdb
            else:
                pdbchain = pdb + chain
        # Prepare the dictionaries:
        if pdbchain not in pdb_to_sequence:
            pdb_to_sequence[pdbchain] = ""
            pdb_to_real_sequence[pdbchain] = ""
            pdb_to_probability[pdbchain] = []
        # Loop through map:
        for n in range(previous_count, previous_count + count):
            if old_datasetmap:
                idx = i
            else:
                idx = n

            pred = list(prediction_matrix[idx])
            curr_res = res_dic[max_idx[idx]]
            pdb_to_probability[pdbchain].append(pred)
            pdb_to_sequence[pdbchain] += curr_res
            if old_datasetmap:
                pdb_to_real_sequence[pdbchain] += res_to_r_dic[res]
        if not old_datasetmap:
            previous_count += count

    if is_consensus:
        last_pdb = ""
        # Sum up probabilities:
        for pdb in pdb_to_sequence.keys():
            curr_pdb = pdb.split("_")[0]
            if last_pdb != curr_pdb:
                pdb_to_consensus_prob[curr_pdb] = np.array(pdb_to_probability[pdb])
                last_pdb = curr_pdb
            else:
                pdb_to_consensus_prob[curr_pdb] = (
                    pdb_to_consensus_prob[curr_pdb] + np.array(pdb_to_probability[pdb])
                ) / 2
        # Extract sequences from consensus probabilities:
        for pdb in pdb_to_consensus_prob.keys():
            pdb_to_consensus[pdb] = ""
            curr_prob = pdb_to_consensus_prob[pdb]
            max_idx = np.argmax(curr_prob, axis=1)
            for m in max_idx:
                curr_res = res_dic[m]
                pdb_to_consensus[pdb] += curr_res

        return (
            pdb_to_sequence,
            pdb_to_probability,
            pdb_to_real_sequence,
            pdb_to_consensus,
            pdb_to_consensus_prob,
        )
    else:
        return pdb_to_sequence, pdb_to_probability, pdb_to_real_sequence, None, None


def save_outputs_to_file(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flat_dataset_map: t.List[t.Tuple],
    model: int,
    model_name: str,
):
    """
    Saves predictions for a specific model to file.

    Parameters
    ----------
    y_true: np.ndarray
        Numpy array of labels (int) 0 or 1.
    y_pred: np.ndarray
        Numpy array of predictions (float) range 0 - 1
    flat_dataset_map: t.List[t.Tuple]
        List of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    model: int
        Number of the model being used.
    model_name: int
        Name of the model being used.
    """
    # Save dataset map only at the beginning:
    if model == 0:
        with open("../encoded_labels.csv", "a") as f:
            y_true = np.asarray(y_true)
            np.savetxt(f, y_true, delimiter=",", fmt="%i")
    flat_dataset_map = np.asarray(flat_dataset_map)
    # Save dataset map only at the beginning:
    if Path("../datasetmap.txt").exists() == False:
        with open("../datasetmap.txt", "a") as f:
            # Output Dataset Map to txt:
            np.savetxt(f, flat_dataset_map, delimiter=",", fmt="%s")

    predictions = np.array(y_pred[model], dtype=np.float16)
    # Output model predictions:
    with open(f"{model_name}.csv", "a") as f:
        np.savetxt(f, predictions, delimiter=",")


blosum62 = {
    ("W", "F"): 1,
    ("L", "R"): -2,
    ("S", "P"): -1,
    ("V", "T"): 0,
    ("Q", "Q"): 5,
    ("N", "A"): -2,
    ("Z", "Y"): -2,
    ("W", "R"): -3,
    ("Q", "A"): -1,
    ("S", "D"): 0,
    ("H", "H"): 8,
    ("S", "H"): -1,
    ("H", "D"): -1,
    ("L", "N"): -3,
    ("W", "A"): -3,
    ("Y", "M"): -1,
    ("G", "R"): -2,
    ("Y", "I"): -1,
    ("Y", "E"): -2,
    ("B", "Y"): -3,
    ("Y", "A"): -2,
    ("V", "D"): -3,
    ("B", "S"): 0,
    ("Y", "Y"): 7,
    ("G", "N"): 0,
    ("E", "C"): -4,
    ("Y", "Q"): -1,
    ("Z", "Z"): 4,
    ("V", "A"): 0,
    ("C", "C"): 9,
    ("M", "R"): -1,
    ("V", "E"): -2,
    ("T", "N"): 0,
    ("P", "P"): 7,
    ("V", "I"): 3,
    ("V", "S"): -2,
    ("Z", "P"): -1,
    ("V", "M"): 1,
    ("T", "F"): -2,
    ("V", "Q"): -2,
    ("K", "K"): 5,
    ("P", "D"): -1,
    ("I", "H"): -3,
    ("I", "D"): -3,
    ("T", "R"): -1,
    ("P", "L"): -3,
    ("K", "G"): -2,
    ("M", "N"): -2,
    ("P", "H"): -2,
    ("F", "Q"): -3,
    ("Z", "G"): -2,
    ("X", "L"): -1,
    ("T", "M"): -1,
    ("Z", "C"): -3,
    ("X", "H"): -1,
    ("D", "R"): -2,
    ("B", "W"): -4,
    ("X", "D"): -1,
    ("Z", "K"): 1,
    ("F", "A"): -2,
    ("Z", "W"): -3,
    ("F", "E"): -3,
    ("D", "N"): 1,
    ("B", "K"): 0,
    ("X", "X"): -1,
    ("F", "I"): 0,
    ("B", "G"): -1,
    ("X", "T"): 0,
    ("F", "M"): 0,
    ("B", "C"): -3,
    ("Z", "I"): -3,
    ("Z", "V"): -2,
    ("S", "S"): 4,
    ("L", "Q"): -2,
    ("W", "E"): -3,
    ("Q", "R"): 1,
    ("N", "N"): 6,
    ("W", "M"): -1,
    ("Q", "C"): -3,
    ("W", "I"): -3,
    ("S", "C"): -1,
    ("L", "A"): -1,
    ("S", "G"): 0,
    ("L", "E"): -3,
    ("W", "Q"): -2,
    ("H", "G"): -2,
    ("S", "K"): 0,
    ("Q", "N"): 0,
    ("N", "R"): 0,
    ("H", "C"): -3,
    ("Y", "N"): -2,
    ("G", "Q"): -2,
    ("Y", "F"): 3,
    ("C", "A"): 0,
    ("V", "L"): 1,
    ("G", "E"): -2,
    ("G", "A"): 0,
    ("K", "R"): 2,
    ("E", "D"): 2,
    ("Y", "R"): -2,
    ("M", "Q"): 0,
    ("T", "I"): -1,
    ("C", "D"): -3,
    ("V", "F"): -1,
    ("T", "A"): 0,
    ("T", "P"): -1,
    ("B", "P"): -2,
    ("T", "E"): -1,
    ("V", "N"): -3,
    ("P", "G"): -2,
    ("M", "A"): -1,
    ("K", "H"): -1,
    ("V", "R"): -3,
    ("P", "C"): -3,
    ("M", "E"): -2,
    ("K", "L"): -2,
    ("V", "V"): 4,
    ("M", "I"): 1,
    ("T", "Q"): -1,
    ("I", "G"): -4,
    ("P", "K"): -1,
    ("M", "M"): 5,
    ("K", "D"): -1,
    ("I", "C"): -1,
    ("Z", "D"): 1,
    ("F", "R"): -3,
    ("X", "K"): -1,
    ("Q", "D"): 0,
    ("X", "G"): -1,
    ("Z", "L"): -3,
    ("X", "C"): -2,
    ("Z", "H"): 0,
    ("B", "L"): -4,
    ("B", "H"): 0,
    ("F", "F"): 6,
    ("X", "W"): -2,
    ("B", "D"): 4,
    ("D", "A"): -2,
    ("S", "L"): -2,
    ("X", "S"): 0,
    ("F", "N"): -3,
    ("S", "R"): -1,
    ("W", "D"): -4,
    ("V", "Y"): -1,
    ("W", "L"): -2,
    ("H", "R"): 0,
    ("W", "H"): -2,
    ("H", "N"): 1,
    ("W", "T"): -2,
    ("T", "T"): 5,
    ("S", "F"): -2,
    ("W", "P"): -4,
    ("L", "D"): -4,
    ("B", "I"): -3,
    ("L", "H"): -3,
    ("S", "N"): 1,
    ("B", "T"): -1,
    ("L", "L"): 4,
    ("Y", "K"): -2,
    ("E", "Q"): 2,
    ("Y", "G"): -3,
    ("Z", "S"): 0,
    ("Y", "C"): -2,
    ("G", "D"): -1,
    ("B", "V"): -3,
    ("E", "A"): -1,
    ("Y", "W"): 2,
    ("E", "E"): 5,
    ("Y", "S"): -2,
    ("C", "N"): -3,
    ("V", "C"): -1,
    ("T", "H"): -2,
    ("P", "R"): -2,
    ("V", "G"): -3,
    ("T", "L"): -1,
    ("V", "K"): -2,
    ("K", "Q"): 1,
    ("R", "A"): -1,
    ("I", "R"): -3,
    ("T", "D"): -1,
    ("P", "F"): -4,
    ("I", "N"): -3,
    ("K", "I"): -3,
    ("M", "D"): -3,
    ("V", "W"): -3,
    ("W", "W"): 11,
    ("M", "H"): -2,
    ("P", "N"): -2,
    ("K", "A"): -1,
    ("M", "L"): 2,
    ("K", "E"): 1,
    ("Z", "E"): 4,
    ("X", "N"): -1,
    ("Z", "A"): -1,
    ("Z", "M"): -1,
    ("X", "F"): -1,
    ("K", "C"): -3,
    ("B", "Q"): 0,
    ("X", "B"): -1,
    ("B", "M"): -3,
    ("F", "C"): -2,
    ("Z", "Q"): 3,
    ("X", "Z"): -1,
    ("F", "G"): -3,
    ("B", "E"): 1,
    ("X", "V"): -1,
    ("F", "K"): -3,
    ("B", "A"): -2,
    ("X", "R"): -1,
    ("D", "D"): 6,
    ("W", "G"): -2,
    ("Z", "F"): -3,
    ("S", "Q"): 0,
    ("W", "C"): -2,
    ("W", "K"): -3,
    ("H", "Q"): 0,
    ("L", "C"): -1,
    ("W", "N"): -4,
    ("S", "A"): 1,
    ("L", "G"): -4,
    ("W", "S"): -3,
    ("S", "E"): 0,
    ("H", "E"): 0,
    ("S", "I"): -2,
    ("H", "A"): -2,
    ("S", "M"): -1,
    ("Y", "L"): -1,
    ("Y", "H"): 2,
    ("Y", "D"): -3,
    ("E", "R"): 0,
    ("X", "P"): -2,
    ("G", "G"): 6,
    ("G", "C"): -3,
    ("E", "N"): 0,
    ("Y", "T"): -2,
    ("Y", "P"): -3,
    ("T", "K"): -1,
    ("A", "A"): 4,
    ("P", "Q"): -1,
    ("T", "C"): -1,
    ("V", "H"): -3,
    ("T", "G"): -2,
    ("I", "Q"): -3,
    ("Z", "T"): -1,
    ("C", "R"): -3,
    ("V", "P"): -2,
    ("P", "E"): -1,
    ("M", "C"): -1,
    ("K", "N"): 0,
    ("I", "I"): 4,
    ("P", "A"): -1,
    ("M", "G"): -3,
    ("T", "S"): 1,
    ("I", "E"): -3,
    ("P", "M"): -2,
    ("M", "K"): -1,
    ("I", "A"): -1,
    ("P", "I"): -3,
    ("R", "R"): 5,
    ("X", "M"): -1,
    ("L", "I"): 2,
    ("X", "I"): -1,
    ("Z", "B"): 1,
    ("X", "E"): -1,
    ("Z", "N"): 0,
    ("X", "A"): 0,
    ("B", "R"): -1,
    ("B", "N"): 3,
    ("F", "D"): -3,
    ("X", "Y"): -1,
    ("Z", "R"): 0,
    ("F", "H"): -1,
    ("B", "F"): -3,
    ("F", "L"): 0,
    ("X", "Q"): -1,
    ("B", "B"): 4,
}