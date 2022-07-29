import argparse
from pathlib import Path
import shutil


def main(args):
    args.input_fasta = Path(args.input_fasta)
    args.input_af2 = Path(args.input_af2)
    assert args.input_fasta.exists(), f"Input file {args.input_fasta} does not exist"
    assert args.input_af2.exists(), f"Input file {args.input_af2} does not exist"
    # all_fasta_paths = list(args.input_fasta.glob('*.fasta'))
    all_pdb_paths = list(args.input_af2.glob('**/*.pdb'))
    for pdb_path in all_pdb_paths:
        fasta_code = args.input_fasta / (pdb_path.parent.name + ".fasta")
        pdb_name = pdb_path.stem
        if fasta_code.exists():
            with open(fasta_code, "r") as f:
                file = f.readlines()
                fasta_name = file[0].strip(">").strip("\n") + "_" + pdb_name + ".pdb"
            shutil.copy2(pdb_path, args.input_af2 / fasta_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_fasta", type=str, help="Path to input file")
    parser.add_argument("--input_af2", type=str, help="Path to af2 input file")
    params = parser.parse_args()
    main(params)
