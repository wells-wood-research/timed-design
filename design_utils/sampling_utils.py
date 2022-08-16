import json
import typing as t
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from ampal.amino_acids import standard_amino_acids

from design_utils.analyse_utils import calculate_seq_metrics


def save_as(pdb_to_sampled: dict, filename: str, mode: str):
    """
    Saves sampled sequences as fasta, json or both.

    Parameters
    ----------
    pdb_to_sampled: dict
        Dictionary {pdb: sampled_sequence}
    filename: str
        Name of file output
    mode: str
        Whether to save in fasta, json or both
    """
    output_paths = []
    print(f"Saving sampled sequences in mode {mode}")
    if mode != "fasta":
        outfile_path = f"{filename}.json"
        output_paths.append(outfile_path)
        with open(outfile_path, "w") as outfile:
            json.dump(pdb_to_sampled, outfile)
    if mode != "json":
        outfile_path = f"{filename}.fasta"
        output_paths.append(outfile_path)
        with open(outfile_path, "w") as outfile:
            for pdb, seq_list in pdb_to_sampled.items():
                for i, seq in enumerate(seq_list):
                    outfile.write(f">{pdb}_{i}\n")
                    outfile.write(f"{seq[0]}\n")  # the first item is the seq
    print("Saving Metrics")
    outfile_path = f"{filename}_metrics.csv"
    output_paths.append(outfile_path)
    with open(outfile_path, "w") as outfile:
        outfile.write(
            "pdb,sequence,charge,isoelectric_point,molecular_weight,molar_extinction\n"
        )
        for pdb, seq_list in pdb_to_sampled.items():
            for i, seq in enumerate(seq_list):
                outfile.write(f"{pdb},{seq[0]},{seq[1]},{seq[2]},{seq[3]},{seq[4]}\n")
    return output_paths


def random_choice_prob_index(
    probs: np.ndarray,
    axis: int = 1,
    return_seq: bool = True,
    rotamer_categories: t.Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Samples from a probability distribution and returns a sequence or the indeces sampled.

    Code adapted from: https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a?noredirect=1&lq=1

    Parameters
    ----------
    probs: np.ndarray
        2D Array with shape (n, n_categries) where n is the number of residues.
    axis: int
        Axis along which to select.
    return_seq: bool
        Whether to return a residue sequence (True) or the index (False)
    rotamer_categories: t.Optional[np.ndarray, None]
        Optionally rotamer categories can be used for sampling.

    Returns
    -------
    sequence: np.ndarray
        Sequence of residues or indeces sampled from a distribution

    """
    r = np.expand_dims(np.random.rand(probs.shape[1 - axis]), axis=axis)
    idxs = (probs.cumsum(axis=axis) > r).argmax(axis=axis)
    if return_seq:
        if rotamer_categories:
            res = np.array(rotamer_categories)
        else:
            res = np.array(list(standard_amino_acids.keys()))
        return res[idxs]
    else:
        return idxs


def sample_from_sequences(
    pdb: str,
    sample_n: int,
    pdb_to_probability: dict,
    rotamer_categories: t.Optional[np.ndarray],
) -> dict:
    """
    Sample from pdb sequences sample_n times.

    Parameters
    ----------
    pdb: str
        pdb to sample from
    sample_n: int
        Number of samples to be drawn
    pdb_to_probability: dict
        Dict {pdb: probability_distribution}
    rotamer_categories: t.Optional[np.ndarray, None]
        Whether to sample from a rotamer distribution or not.

    Returns
    -------
    pdb_to_sample: dict
        Dict {pdb: [(n, n_sample)]}

    """
    pdb_to_sample = {}
    # Sample from distribution
    sampled_seq_list = []
    # TODO parallelize:
    for i in range(sample_n):
        seq_list = random_choice_prob_index(
            np.array(pdb_to_probability[pdb]),
            return_seq=True,
            rotamer_categories=rotamer_categories,
        )
        # Join seq from residue list to one string
        sampled_seq = "".join(seq_list)
        # Calculate sequence metrics
        metrics_tuple = calculate_seq_metrics(sampled_seq)
        sampled_seq_list.append((sampled_seq, *metrics_tuple))
    pdb_to_sample[pdb] = sampled_seq_list

    return pdb_to_sample


def apply_temp_to_probs(probs: np.ndarray, t: int = 1.0):
    """
    Applies a temperature factor to a softmax output probability.

    Adapted from https://stackoverflow.com/questions/37246030/how-to-change-the-temperature-of-a-softmax-output-in-keras

    Parameters
    ----------
    probs: np.ndarray
        2D Probability Array
    t: float
        Temperature Factor


    Returns
    -------
    probs: np.ndarray
        2D Probability Array with t applied to it.

    """
    probs = np.array(probs) ** (1 / t)
    p_sum = np.sum(probs, axis=1)
    return probs / p_sum[:, None]


def sample_with_multiprocessing(
    workers, pdb_codes, sample_n, pdb_to_probability, flat_categories
):
    """

    Parameters
    ----------
    workers
    pdb_codes
    sample_n
    pdb_to_probability
    flat_categories

    Returns
    -------

    """
    with Pool(processes=workers) as p:
        pdb_to_sample_dict_list = p.starmap(
            sample_from_sequences,
            zip(
                pdb_codes,
                repeat(sample_n),
                repeat(pdb_to_probability),
                repeat(flat_categories),
            ),
        )
        p.close()
    # Flatten dictionary:
    pdb_to_sample = {}
    for curr_dict in pdb_to_sample_dict_list:
        if curr_dict is not None:
            pdb_to_sample.update(curr_dict)
    return pdb_to_sample
