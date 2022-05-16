import argparse
from pathlib import Path

import numpy as np


def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    # Loop through each model and temp:
    for model in args.models:
        output_dict = {}
        for t in args.temperature:
            # Compose metrics and solubility files:
            metrics_file = args.input_path / f"{model}_temp_{t}_n_200_metrics.csv" # TODO: deal with original n and sample n
            # Check paths exist:
            assert metrics_file.exists(), f"Metrics file {metrics_file} does not exist."
            # Load files into an array:
            metrics_arr = np.genfromtxt(metrics_file, delimiter=",", dtype=str)
            pdb_codes = np.unique(metrics_arr[:, 0])[:args.pdb_n]
            for pdb in pdb_codes:
                sliced_arr = metrics_arr[metrics_arr[:, 0] == pdb]
                for i, line in enumerate(sliced_arr[:args.sample_n, :]):
                    _, seq, _, _ = line
                    pdb_key = f"{model}_{pdb}_{t}_{i}"
                    output_dict[pdb_key] = seq
        file_count = 1
        # Create output folder if it does not exist:
        output_path = Path(f"{model}_{file_count}")
        output_path.mkdir(parents=True, exist_ok=True)
        for i, (pdb, seq) in enumerate(output_dict.items()):
            # Split when group has enough sequences:
            if i == file_count*args.structures_per_category:
                file_count += 1
                output_path = Path(f"{model}_{file_count}")
                # Create output folder if it does not exist:
                output_path.mkdir(parents=True, exist_ok=True)
            # TODO: there are better ways to deal with open files
            with open(f"{output_path}/{model}_{i}.fasta", "a+") as f:
                f.write(f">{pdb}\n{seq}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--models",
        type=list,
        nargs="+",
        default=[
            "TIMED_rot",
            "TIMED",
        ],
        help="Which models to analyse (default: TIMED_rot, TIMED).",
    )
    parser.add_argument(
        "--temperature",
        type=list,
        nargs="+",
        default=[0.1, 0.5, 2.0, 5.0],
        help="Which Temperatures to analyse (default: 0.1, 0.5, 2.0, 5.0).",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=15,
        help="Number of samples to select (default: 15).",
    )
    parser.add_argument(
        "--pdb_n",
        type=int,
        default=10,
        help="Number of pdbs to select (default: 10).",
    )
    parser.add_argument(
        "--structures_per_category",
        type=int,
        default=80,
        help="Number of structures to output per file (default: 80).",
    )
    params = parser.parse_args()
    main(params)
