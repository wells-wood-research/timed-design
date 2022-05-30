import argparse
from pathlib import Path

from utils.utils import load_dataset_and_predict


def main(args):
    args.path_to_dataset = Path(args.path_to_dataset)
    args.path_to_model = Path(args.path_to_model)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
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
        "--output_analysis",
        type=bool,
        action='store_true',
        help="Whether to output analysis graphs.",
    )
    parser.add_argument(
        "--predict_rotamers",
        type=bool,
        action='store_true',
        help="Whether model outputs predictions for 338 rotamers (True) or 20 residues (False).",
    )
    params = parser.parse_args()
    main(params)
