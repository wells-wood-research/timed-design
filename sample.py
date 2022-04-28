import argparse
from pathlib import Path

import numpy as np

from utils.analyse_utils import calculate_seq_metrics
from utils.sampling_utils import (
    random_choice_prob_index,
    save_as,
)
from utils.utils import extract_sequence_from_pred_matrix, get_rotamer_codec


def main(args):
    args.path_to_pred_matrix = Path(args.path_to_pred_matrix)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    assert (
        args.path_to_pred_matrix.exists()
    ), f"Prediction Matrix file {args.path_to_pred_matrix} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Dataset Map file {args.path_to_datasetmap} does not exist"
    prediction_matrix = np.genfromtxt(
        args.path_to_pred_matrix, delimiter=",", dtype=np.float16
    )
    dataset_map = np.genfromtxt(
        args.path_to_datasetmap,
        delimiter=",",
        dtype=str,
    )
    if args.predict_rotamers:
        codec, flat_categories = get_rotamer_codec()
    else:
        codec, flat_categories = None, None
    # TODO: Load structures if available
    # TODO: sample from rotamer
    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        _,
        _,
    ) = extract_sequence_from_pred_matrix(
        dataset_map, prediction_matrix, rotamers_categories=flat_categories
    )
    # TODO add multiprocessing:
    pdb_to_sample = {}
    print(
        f"Ready to sample {args.sample_n} for each of the {len(list(pdb_to_probability.keys()))} proteins."
    )
    for pdb in pdb_to_probability.keys():
        # Sample from distribution
        sampled_seq_list = []
        for i in range(args.sample_n):
            seq_list = random_choice_prob_index(np.array(pdb_to_probability[pdb]))
            # Join seq from residue list to one string
            sampled_seq = "".join(seq_list)
            # Calculate sequence metrics
            charge, iso_ph = calculate_seq_metrics(sampled_seq)
            sampled_seq_list.append((sampled_seq,charge, iso_ph))
        pdb_to_sample[pdb] = sampled_seq_list
    # Save sequences to files:
    save_as(pdb_to_sample, model_name=args.path_to_pred_matrix.stem, mode=args.save_as)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path_to_pred_matrix",
        type=str,
        help="Path to prediction matrix file ending with .csv",
    )
    parser.add_argument(
        "--path_to_datasetmap",
        default="datasetmap.txt",
        type=str,
        help="Path to dataset map ending with .txt",
    )
    parser.add_argument(
        "--predict_rotamers",
        type=bool,
        default=False,
        help="Whether model outputs predictions for 338 rotamers (True) or 20 residues (False).",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=100,
        help="Number of samples to be drawn from the distribution.",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        default="all",
        const="all",
        nargs="?",
        choices=["fasta", "json", "all"],
        help="Whether to save as fasta and json (default: all) or either of them.",
    )
    params = parser.parse_args()
    main(params)
