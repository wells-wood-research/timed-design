import argparse
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from utils.sampling_utils import (sample_from_sequences, save_as)
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
    pdb_codes = list(pdb_to_probability.keys())
    print(
        f"Ready to sample {args.sample_n} for each of the {len(pdb_codes)} proteins."
    )
    with Pool(processes=args.workers) as p:
        pdb_to_sample_dict_list = p.starmap(
            sample_from_sequences,
            zip(
                pdb_codes,
                repeat(args.sample_n),
                repeat(pdb_to_probability),
            ),
        )
        p.close()
    # Flatten dictionary:
    pdb_to_sample = {}
    for curr_dict in pdb_to_sample_dict_list:
        if curr_dict is not None:
            pdb_to_sample.update(curr_dict)
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
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    params = parser.parse_args()
    main(params)