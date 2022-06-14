import argparse
from pathlib import Path

import numpy as np


def main(args):
    args.input_path = Path(args.input_path)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    # Loop through each model:
    for model in args.models:
        model_path = args.input_path / model
        assert model_path.exists(), f"Model Path {args.model_path} does not exist"
        all_files = list(model_path.glob("**/*"))
        fasta_paths = ""
        for i in range(1, len(all_files)+1):
            base_path = f"$PWD/rds/rds-bmai-cdt-VCjWBHEJ998/publication_uncertainty/{model}/{model}_{i}.fasta,"
            fasta_paths += base_path
            if i % args.structures_per_script == 0:
                with open(f"af_{model}_{i // args.structures_per_script}.sh", "w") as f:
                    script = f"""#!/bin/bash
#SBATCH -A BMAI-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL

# load appropriate modules
source ~/rds/hpc-work/conda/bin/conda
source ~/.bashrc
source ~/.bash_profile
module load rhel8/default-amp
module load alphafold

# point to location of AlphaFold data
DATA=/rds/project/rds-dMMtPvqHWv4/data/

run_alphafold \\
--pdb70_database_path=/data/pdb70/pdb70 \\
--bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \\
--uniclust30_database_path /data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \\
--output_dir $PWD/rds/rds-bmai-cdt-VCjWBHEJ998/publication_uncertainty/output/{model} \\
--fasta_paths {fasta_paths} \\
--max_template_date=2020-05-14 \\
--db_preset=full_dbs \\
--use_gpu_relax=True \\
--cpus 32
                    """
                    f.write(script)
                fasta_paths = ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to input file")
    parser.add_argument(
        "--models",
        type=list,
        nargs="+",
        default=[
            "TIMED_Deep",
            "TIMED_rotamer_balanced_rot",
            "TIMED_rotamer_deep_rot",
            "TIMED_rotamer_not_so_deep_rot",
            "TIMED_rotamer_rot",
            "TIMED",
        ],
        help="Which models to analyse (default: all).",
    )
    parser.add_argument(
        "--structures_per_script",
        type=int,
        default=80,
        help="Number of structures to output per file (default: 80).",
    )
    params = parser.parse_args()
    main(params)
