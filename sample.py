import argparse
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ampal.amino_acids import standard_amino_acids

from utils.sampling_utils import apply_temp_to_probs, sample_from_sequences, \
    save_as
from utils.utils import extract_sequence_from_pred_matrix, get_rotamer_codec


def main(args):
    # Set Random seed:
    np.random.default_rng(seed=args.seed)
    # Sanitise Paths:
    args.path_to_pred_matrix = Path(args.path_to_pred_matrix)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    assert (
        args.path_to_pred_matrix.exists()
    ), f"Prediction Matrix file {args.path_to_pred_matrix} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Dataset Map file {args.path_to_datasetmap} does not exist"
    # Load prediction matrix:
    prediction_matrix = np.genfromtxt(
        args.path_to_pred_matrix, delimiter=",", dtype=np.float64
    )
    # Load dataset map:
    if args.support_old_datasetmap:
        dataset_map = np.genfromtxt(
            args.path_to_datasetmap,
            delimiter=",",
            dtype=str,
        )
    else:
        dataset_map = np.genfromtxt(
            args.path_to_datasetmap,
            delimiter=" ",
            dtype=str,
            skip_header=3,
        )
    # Apply temperature factor to prediction matrix:
    if args.temperature != 1:
        prediction_matrix = apply_temp_to_probs(prediction_matrix, t=args.temperature)
    # Load codec:
    if args.predict_rotamers:
        # Get rotamer categories:
        _, flat_categories = get_rotamer_codec()
        # Get dictionary for 3 letter -> 1 letter conversion:
        res_to_r = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
        # Create flat categories of 1 letter amino acid for each of the 338 rotamers:
        flat_categories = [res_to_r[res.split("_")[0]] for res in flat_categories]
        # Extract dictionaries with sequences:
    else:
        _, flat_categories = None, None

    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        _,
        _,
    ) = extract_sequence_from_pred_matrix(
        dataset_map,
        prediction_matrix,
        rotamers_categories=flat_categories,
        old_datasetmap=args.support_old_datasetmap,
    )
    # Select only 59 structures used for sampling:
    af2_benchmark_structures = ["1hq0A", "1a41A", "1ds1A", "1dvoA", "1g3pA",
                                "1h70A", "1hxrA", "1jovA", "1l0sA", "1o7iA",
                                "1uzkA", "1x8qA", "2bhuA", "2dyiA", "2imhA",
                                "2j8kA", "2of3A", "2ra1A", "2v3gA", "2v3iA",
                                "2w18A", "3cxbA", "3dadA", "3dkrA", "3e3vA",
                                "3e4gA", "3e8tA", "3essA", "3giaA", "3gohA",
                                "3hvmA", "3klkA", "3kluA", "3kstA", "3kyfA",
                                "3maoA", "3o4pA", "3oajA", "3q1nA", "3rf0A",
                                "3swgA", "3zbdA", "3zh4A", "4a6qA", "4ecoA",
                                "4efpA", "4fcgA", "4fs7A", "4i1kA", "4le7A",
                                "4m4dA", "4ozwA", "4wp6A", "4y5jA", "5b1rA",
                                "5bufA", "5c12A", "5dicA", "6baqA"]
    # TODO: Improve implementation
    pdb_codes = af2_benchmark_structures
    print(f"Ready to sample {args.sample_n} for each of the {len(pdb_codes)} proteins.")
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
    save_as(
        pdb_to_sample,
        filename=f"{args.path_to_pred_matrix.stem}_temp_{args.temperature}_n_{args.sample_n}",
        mode=args.save_as,
    )


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
        default=False,
        action="store_true",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Temperature factor to apply to softmax prediction. (default: 1.0 - unchanged)",
    )
    parser.add_argument(
        "--support_old_datasetmap",
        default=False,
        action="store_true",
        help="Whether model to import from the old datasetmap (default: False)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    params = parser.parse_args()
    main(params)
