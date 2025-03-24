import gzip, sys, random, string
import typing as t
import warnings
from collections import defaultdict
from pathlib import Path

import ampal
import h5py
import numpy as np
from ampal.amino_acids import (
    standard_amino_acids,
    polarity_Zimmerman,
    residue_charge,
)

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


def create_flat_dataset_map(
    frame_dataset: Path,
) -> (t.List[t.Tuple[str, int, str, str]], t.Set[str]):
    """
    Flattens the structure of the h5 dataset for batching and balancing purposes.

    Parameters
    ----------
    frame_dataset: Path
        Path to the .hdf5 dataset with a hierarchical structure of PDB codes, chains, and residues.

    Returns
    -------
    flat_dataset_map: List[Tuple[str, str, str, str]]
        Flattened dataset structure containing (pdb_code, chain_id, residue_id, residue_label).
    pdb_set: Set[str]
        Set of all PDB codes included in the dataset
    """
    standard_residues = set(standard_amino_acids.values())  # Use a set for O(1) lookups
    flat_dataset_map = []
    pdbs_set = set()

    with h5py.File(frame_dataset, "r") as dataset_file:
        for pdb_code, pdb_group in dataset_file.items():
            for chain_id, chain_group in pdb_group.items():
                for residue_id, residue_group in chain_group.items():
                    residue_label = residue_group.attrs["label"]

                    if residue_label not in standard_residues:
                        # Convert uncommon residue if applicable
                        new_label = UNCOMMON_RESIDUE_DICT.get(residue_label)
                        if new_label:
                            warnings.warn(
                                f"{residue_label} is not standard; converted to {new_label}."
                            )
                            residue_label = new_label
                        else:
                            raise ValueError(
                                f"Unexpected residue label: {residue_label}"
                            )

                    flat_dataset_map.append(
                        (pdb_code, chain_id, residue_id, residue_label)
                    )

                pdbs_set.add(pdb_code)  # Add after processing the chain

    return flat_dataset_map, pdbs_set


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
    flat_dataset_map: np.ndarray,
    path_to_benchmark_map: Path,
):
    """
    Converts datasetmap for compatibility with PDBench / Sequence recovery benchmark

    Parameters
    ----------
    flat_dataset_map: np.ndarray
        Dataset map array
    path_to_benchmark_map: Path
        Path to the benchmark dataset map file
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

    with open(path_to_benchmark_map, "w") as d:
        d.write("ignore_uncommon False\ninclude_pdbs\n##########\n")
        for pdb, count in count_dict.items():
            d.write(f"{pdb} {count}\n")


def save_dict_to_fasta(
    pdb_to_sequence: dict,
    model_name: str,
    path_to_output: Path = Path.cwd(),
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
    flat_dataset_map: np.ndarray,
    prediction_matrix: np.ndarray,
) -> t.Tuple[t.Dict[str, t.Dict[str, str]], t.Dict[str, np.ndarray]]:
    """
    Extract sequence from prediction matrix and create pdb_to_sequence and
    pdb_to_probability dictionaries

    Parameters
    ----------
    flat_dataset_map: np.ndarray
        Array of tuples with the order
        [... (pdb_code, chain_id, residue_id,  residue_label, encoded_residue) ...]
    prediction_matrix: np.ndarray
        Prediction matrix for each of the sequence

    Returns
    -------
    pdb_to_sequence: dict
        Dictionary {pdb_code: {'wildtype': wildtype_sequence, 'argmax': argmax_sequence}}
    pdb_to_probability: dict
        Dictionary {pdb_code: probability}
    """
    pdb_sequences = defaultdict(lambda: {'wildtype': '', 'argmax': ''})
    pdb_to_probability = defaultdict(list)

    res_to_r_dic = {v: k for k, v in standard_amino_acids.items()}
    res_dic = list(standard_amino_acids.keys())
    max_idx = np.argmax(prediction_matrix, axis=1)

    for i, (pdb, chain, _, res) in enumerate(flat_dataset_map):
        pdb_chain = pdb + chain
        pred = prediction_matrix[i].tolist()
        curr_res = res_dic[max_idx[i]]

        pdb_to_probability[pdb_chain].append(pred)
        pdb_sequences[pdb_chain]['argmax'] += curr_res
        pdb_sequences[pdb_chain]['wildtype'] += res_to_r_dic[res]

    return dict(pdb_sequences), dict(pdb_to_probability)


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
