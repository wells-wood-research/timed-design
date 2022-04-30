import json

import numpy as np
from ampal.amino_acids import standard_amino_acids

from utils.analyse_utils import calculate_seq_metrics


def save_as(pdb_to_sampled, model_name, mode):
    """
    Saves sampled sequences as fasta, json or both.

    Parameters
    ----------
    pdb_to_sampled
    model_name
    mode

    Returns
    -------

    """
    print(f"Saving sampled sequences in mode {mode}")
    if mode != "fasta":
        with open(f"{model_name}.json", "w") as outfile:
            json.dump(pdb_to_sampled, outfile)
    if mode != "json":
        with open(f"{model_name}.fasta", "w") as outfile:
            for pdb, seq_list in pdb_to_sampled.items():
                for i, seq in enumerate(seq_list):
                    outfile.write(f">{pdb}_{i}\n")
                    outfile.write(f"{seq[0]}\n")  # the first item is the seq
    print("Saving Metrics")
    with open(f"{model_name}_metrics.csv", "w") as outfile:
        for pdb, seq_list in pdb_to_sampled.items():
            for i, seq in enumerate(seq_list):
                outfile.write(f"{pdb},{seq[0]},{seq[1]},{seq[2]}\n")


def random_choice_prob_index(probs, axis=1, return_seq=True):
    """
    Code adapted from: https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a?noredirect=1&lq=1

    Parameters
    ----------
    probs
    axis
    return_seq

    Returns
    -------

    """
    r = np.expand_dims(np.random.rand(probs.shape[1 - axis]), axis=axis)
    idxs = (probs.cumsum(axis=axis) > r).argmax(axis=axis)
    if return_seq:
        res = np.array(list(standard_amino_acids.keys()))
        return res[idxs]
    else:
        return idxs


def sample_from_sequences(pdb, sample_n, pdb_to_probability):
    pdb_to_sample = {}
    # Sample from distribution
    sampled_seq_list = []
    # TODO parallelize:
    for i in range(sample_n):
        seq_list = random_choice_prob_index(np.array(pdb_to_probability[pdb]))
        # Join seq from residue list to one string
        sampled_seq = "".join(seq_list)
        # Calculate sequence metrics
        charge, iso_ph = calculate_seq_metrics(sampled_seq)
        sampled_seq_list.append((sampled_seq, charge, iso_ph))
    pdb_to_sample[pdb] = sampled_seq_list

    return pdb_to_sample
