import gzip, sys, random, string
import typing as t
import warnings
from itertools import product
from pathlib import Path

import ampal
import h5py
import numpy as np
from ampal.amino_acids import (
    side_chain_dihedrals,
    standard_amino_acids,
    polarity_Zimmerman,
    residue_charge,
)
from numpy import genfromtxt

from aposteriori.config import MAKE_FRAME_DATASET_VER, UNCOMMON_RESIDUE_DICT
from aposteriori.data_prep.create_frame_data_set import DatasetMetadata


def rm_tree(pth: Path):
    # Removes all files in a directory and the directory. From https://stackoverflow.com/questions/50186904/pathlib-recursively-remove-directory
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def load_pdb_from_path(structure_path: Path) -> ampal.Assembly:
    """
    Simple utility to load PDB file into ampal and deal with .gz / containers

    Parameters
    ----------
    structure_path: Path
        Path to PDB structure

    Returns
    -------
    pdb_structure: ampal.Assembly
        Ampal assembly for structure path

    """
    # Load structure:
    if structure_path.suffix == ".gz":
        with gzip.open(str(structure_path), "rb") as inf:
            pdb_structure = ampal.load_pdb(inf.read().decode(), path=False)
    else:
        pdb_structure = ampal.load_pdb(str(structure_path))
    # Select first state of container:
    if isinstance(pdb_structure, ampal.AmpalContainer):
        pdb_structure = pdb_structure[0]
    return pdb_structure


def modify_pdb_with_input_property(
    structure_path: Path, property_map: np.ndarray, property: str
) -> ampal.Assembly:
    """
    Modifies input structure with polarity. A bit hacky.

    Replaces residues letter to be changed to ALA for no polarity and K for polarity.

    Parameters
    ----------
    structure_path: Path
        Path to structures
    property_map: np.ndarray
        Property map

    Returns
    -------
    pdb_structure: ampal.Assembly
        Ampal structure with modified letter code

    """
    property = property.lower()
    accepted_properties = ["polarity", "charge"]
    assert (
        property in accepted_properties
    ), f"Property {property} not found among {accepted_properties}"
    property_dict = {0: "A", 1: "K", -1: "D"}
    pdb_structure = load_pdb_from_path(structure_path)
    count = 0
    merged_sequence = ""
    for chain in pdb_structure:
        for res in chain:
            r = res.mol_letter
            if r in standard_amino_acids.keys():
                if property == "polarity":
                    res_property = 0 if polarity_Zimmerman[r] < 20 else 1
                else:
                    res_property = residue_charge[r]
            else:
                res_property = 0
            if property_map[count] != res_property:
                res.mol_code = standard_amino_acids[property_dict[property_map[count]]]
                res.mol_letter = property_dict[property_map[count]]
            merged_sequence += res.mol_letter
            count += 1
    new_property_map = convert_seq_to_property(merged_sequence, property=property)
    np.testing.assert_array_equal(
        new_property_map, property_map, err_msg="Property maps differ."
    )

    return pdb_structure


def create_residue_map_from_pdb(structure_path: Path) -> (t.List[str], str):
    """
    Creates a residue map (similar to dataset map) based on a pdb file.

    Parameters
    ----------
    structure_path: Path
        Path to pdb structure.

    Returns
    -------
    residue_map: t.List[str]
        Residue map of the form ["{res.mol_letter}{res.id} (Chain {chain.id})" ...]
    merged_sequence: str
        Full sequence merged into one string. If multiple chains, it squashes all the sequences together.
    """
    pdb_structure = load_pdb_from_path(structure_path)
    residue_map = []
    merged_sequence = ""
    for chain in pdb_structure:
        for res in chain:
            residue_map.append(f"{res.mol_letter}{res.id} (Chain {chain.id})")
            merged_sequence += res.mol_letter
    return residue_map, merged_sequence


def convert_seq_to_property(seq: str, property: str) -> t.List[int]:
    """
    Converts sequence of residues into property list from either polarity or charge.

    Parameters
    ----------
    seq: str
        Seq of residues
    property: str
        Property to be encoded

    Returns
    -------
    output: t.List[int]
        List of ints containing property of interest
    """
    accepted_properties = ["polarity", "charge"]
    assert (
        property.lower() in accepted_properties
    ), f"Property {property} not found among {accepted_properties}"
    res_list = list(seq)
    if property == "polarity":
        output_list = []
        for r in res_list:
            if r in standard_amino_acids.keys():
                output_list.append(0 if polarity_Zimmerman[r] < 20 else 1)
            else:
                output_list.append(0)
        return output_list
    else:
        return [residue_charge[r] for r in res_list]


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
    dataset_map = np.asarray(dataset_map)
    # If list only contains 1 pdb, it fails to create a list of list [pdb_code, count]
    if isinstance(dataset_map[0], str):
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
        └─[pdb_code] Contains a number of subgroups, one for each chain.
          └─[chain_id] Contains a number of subgroups, one for each residue.
            └─[residue_id] voxels_per_side^3 array of ints, representing element number.
              └─.attrs['label'] Three-letter code for the residue.
              └─.attrs['encoded_residue'] One-hot encoding of the residue.
        └─.attrs['make_frame_dataset_ver']: str - Version used to produce the dataset.
        └─.attrs['frame_dims']: t.Tuple[int, int, int, int] - Dimentsions of the frame.
        └─.attrs['atom_encoder']: t.List[str] - Lables used for the encoding (eg, ["C", "N", "O"]).
        └─.attrs['encode_cb']: bool - Whether a Cb atom was added at the avg position of (-0.741287356, -0.53937931, -1.224287356).
        └─.attrs['atom_filter_fn']: str - Function used to filter the atoms in the frame.
        └─.attrs['residue_encoder']: t.List[str] - Ordered list of residues corresponding to the encoding used.
        └─.attrs['frame_edge_length']: float - Length of the frame in Angstroms (A)
        └─.attrs['voxels_as_gaussian']: bool - Whether the voxels are encoded as a floating point of a gaussian (True) or boolean (False)


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
        └─[pdb_code] Contains a number of subgroups, one for each chain.
          └─[chain_id] Contains a number of subgroups, one for each residue.
            └─[residue_id] voxels_per_side^3 array of ints, representing element number.
              └─.attrs['label'] Three-letter code for the residue.
              └─.attrs['encoded_residue'] One-hot encoding of the residue.
        └─.attrs['make_frame_dataset_ver']: str - Version used to produce the dataset.
        └─.attrs['frame_dims']: t.Tuple[int, int, int, int] - Dimentsions of the frame.
        └─.attrs['atom_encoder']: t.List[str] - Lables used for the encoding (eg, ["C", "N", "O"]).
        └─.attrs['encode_cb']: bool - Whether a Cb atom was added at the avg position of (-0.741287356, -0.53937931, -1.224287356).
        └─.attrs['atom_filter_fn']: str - Function used to filter the atoms in the frame.
        └─.attrs['residue_encoder']: t.List[str] - Ordered list of residues corresponding to the encoding used.
        └─.attrs['frame_edge_length']: float - Length of the frame in Angstroms (A)
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


def get_rotamer_codec(return_reduction_guide: bool = False) -> t.Optional[t.List[int]]:
    """
    Creates a codec for tagging residues rotamers.

    return_reduction_guide: Bool
        Whether to return reduction guide to squash

    Returns
    -------
    rot_to_20res: dict
        Rotamer residues encoding of the format {rotamer_number : [ (20,) encoding ]}
    flat_categories: t.List[str]
        Categories of rotamers (338,) eg. ['ALA_0', 'CYS_1', 'CYS_2', 'CYS_3', 'ASP_11', 'ASP_12', 'ASP_13', ... ]
    reduction_guide: t.Optional[t.List[str]]
        List of int indicating which idxs to reduce:
        [0, 1, 4, 13, 40, 49, 50, 59, 68, 149, 158, 185, 194, 203, 230, 311, 314, 317, 320, 329]
        https://github.com/wells-wood-research/timed-design/issues/7
    """
    res_rot_to_encoding = {}
    flat_categories = []
    rot_to_20res = {}
    all_count = 338
    r_count = 0  # Number of rotamers processed so far
    reduction_guide = []
    for i, (a, res) in enumerate(standard_amino_acids.items()):
        reduction_guide.append(r_count)
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
    if return_reduction_guide:
        return rot_to_20res, flat_categories, reduction_guide
    else:
        return rot_to_20res, flat_categories


def compress_rotamer_predictions_to_20(prediction_matrix: np.ndarray) -> np.ndarray:
    """
    Converts the rotamer prediction matrix from (n, 388) to (n, 20)

    Parameters
    ----------
    prediction_matrix: np.ndarray
        Rotamer prediction matrix (n, 388)

    Returns
    -------
    reduced_prediction_matrix: np.ndarray
        Reduced rotamer prediction matrix (n, 20)

    """
    _, _, reduction_guide = get_rotamer_codec(return_reduction_guide=True)
    return np.add.reduceat(prediction_matrix, reduction_guide, axis=1)


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


def convert_dataset_map_for_srb(
    flat_dataset_map: list,
    model_name: str,
    path_to_output: Path = Path.cwd(),
):
    """
    Converts datasetmap for compatibility with PDBench / Sequence recovery benchmark

    Parameters
    ----------
    flat_dataset_map: list
        Dataset map list
    model_name: str
        Name of model
    path_to_output: Path
        Path to output directory. Defaults to current working directory.
    """
    count_dict = {}
    for i, (pdb, chain, res_idx, _) in enumerate(flat_dataset_map):
        if "_0" in pdb:
            pdb = pdb.split("_0")[0]
        # Add chain to PDB_code TODO: this is not robust in case the user has 4 letter name. Unsure what's the best way of dealing with this.
        if len(pdb) == 4:
            pdb += chain
        if pdb not in count_dict:
            count_dict[pdb] = 0

        count_dict[pdb] += 1

    path_to_datasetmap = path_to_output / f"{model_name}.txt"
    with open(path_to_datasetmap, "w") as d:
        d.write("ignore_uncommon False\ninclude_pdbs\n##########\n")
        for pdb, count in count_dict.items():
            d.write(f"{pdb} {count}\n")


def save_consensus_probs(
    pdb_to_consensus_prob: dict, model_name: str, path_to_output: Path = Path.cwd()
):
    """
    Saves consensus sequence into PDBench-compatible format.

    Parameters
    ----------
    pdb_to_consensus_prob: dict
        Dictionary {pdb_code: consensus_probabilities}
    model_name: dict
        Name of the model
    path_to_output: Path
        Path to output directory. Defaults to current working directory.

    """
    path_to_consensus = path_to_output / f"{model_name}_consensus.txt"
    with open(path_to_consensus, "w") as d, open(
        f"{model_name}_consensus.csv", "a"
    ) as p:
        d.write("ignore_uncommon False\ninclude_pdbs\n##########\n")
        for pdb, predictions in pdb_to_consensus_prob.items():
            d.write(f"{pdb} {len(predictions)}\n")
            np.savetxt(p, predictions, delimiter=",")


def save_dict_to_fasta(
    pdb_to_sequence: dict, model_name: str, path_to_output: Path = Path.cwd(),
):
    """
    Saves a dictionary of protein sequences to a fasta file.

    Parameters
    ----------
    pdb_to_sequence: dict
        Dictionary {pdb_code: predicted_sequence}
    model_name: str
        Name of the model.
    output_dir: Path
        Path to output directory. Defaults to current working directory.
    """
    path_to_fasta = path_to_output / f"{model_name}.fasta"
    with open(path_to_fasta, "w") as f:
        for pdb, seq in pdb_to_sequence.items():
            f.write(f">{pdb}\n{seq}\n")


def extract_sequence_from_pred_matrix(
    flat_dataset_map: t.List[t.Tuple],
    prediction_matrix: np.ndarray,
    rotamers_categories: t.List[str],
    old_datasetmap: bool = False,
    is_consensus: bool = False,
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
            pdb_chain, chain, _, res = flat_dataset_map[i]
            count = 1
        else:
            pdb_chain, count = flat_dataset_map[i]
            count = int(count)
            chain = ""
        pdb_chain += chain
        # Prepare the dictionaries:
        if pdb_chain not in pdb_to_sequence:
            pdb_to_sequence[pdb_chain] = ""
            pdb_to_real_sequence[pdb_chain] = ""
            pdb_to_probability[pdb_chain] = []
        # Loop through map:
        for n in range(previous_count, previous_count + count):
            if old_datasetmap:
                idx = i
            else:
                idx = n

            pred = list(prediction_matrix[idx])
            curr_res = res_dic[max_idx[idx]]
            pdb_to_probability[pdb_chain].append(pred)
            pdb_to_sequence[pdb_chain] += curr_res
            if old_datasetmap:
                pdb_to_real_sequence[pdb_chain] += res_to_r_dic[res]
        if not old_datasetmap:
            previous_count += count

    if is_consensus:
        last_pdb = ""
        # Sum up probabilities:
        for pdb_chain in pdb_to_sequence.keys():
            curr_pdb = pdb_chain.split("_")[0]
            if last_pdb != curr_pdb:
                pdb_to_consensus_prob[curr_pdb] = np.array(pdb_to_probability[pdb_chain])
                last_pdb = curr_pdb
            else:
                pdb_to_consensus_prob[curr_pdb] = (
                    pdb_to_consensus_prob[curr_pdb] + np.array(pdb_to_probability[pdb_chain])
                ) / 2
        # Extract sequences from consensus probabilities:
        for pdb_chain in pdb_to_consensus_prob.keys():
            pdb_to_consensus[pdb_chain] = ""
            curr_prob = pdb_to_consensus_prob[pdb_chain]
            max_idx = np.argmax(curr_prob, axis=1)
            for m in max_idx:
                curr_res = res_dic[m]
                pdb_to_consensus[pdb_chain] += curr_res

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
    path_to_output: Path = Path.cwd(),
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
    path_to_output: Path
        Path to output directory. Defaults to current working directory.
    """
    path_to_encoded_labels = path_to_output / "encoded_labels.csv"
    path_to_datasetmap = path_to_output / "datasetmap.txt"
    path_to_predictions = path_to_output / f"{model_name}.csv"
    # Save dataset map only at the beginning:
    if model == 0:
        with open(path_to_encoded_labels, "a") as f:
            y_true = np.asarray(y_true)
            np.savetxt(f, y_true, delimiter=",", fmt="%i")
    flat_dataset_map = np.asarray(flat_dataset_map)
    # Save dataset map only at the beginning:
    if path_to_datasetmap.exists() == False:
        with open(path_to_datasetmap, "a") as f:
            # Output Dataset Map to txt:
            np.savetxt(f, flat_dataset_map, delimiter=",", fmt="%s")

    predictions = np.array(y_pred[model], dtype=np.float16)
    # Output model predictions:
    with open(path_to_predictions, "a") as f:
        np.savetxt(f, predictions, delimiter=",")


def create_map_alphanumeric_code(property_map: np.ndarray, k: int = 32) -> str:
    """
    Creates alphanumeric code based on property map

    Parameters
    ----------
    property_map: np.ndarray
        Array of property of length (n_residues,)
    k: int
        Number of characters used in the alphanumeric code

    Returns
    -------
    map_code: str
        String containing k alphanumeric characters
    """
    # Create alphanumeric code based on polarity map:
    seed_map = "1"
    for i in property_map:
        # Dealing with negative charge:
        if i < 0:
            seed_map += str(2)
        else:
            seed_map += str(i)
    seed_map = int(seed_map)
    # Set random seed for repeatability
    random.seed(seed_map)
    # Create alphanumeric code
    map_code = "".join(random.choices(string.ascii_letters + string.digits, k=k))
    return map_code


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
