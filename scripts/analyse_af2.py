import argparse
from pathlib import Path

import numpy as np
import pymol


def calculate_GDT(pdb_original_path, pdb_predicted_path):
    """
    sel1: protein to compare
    sel2: reference protein
    """
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    cmd.delete("all")
    cmd.load(pdb_original_path)
    cmd.load(pdb_predicted_path)
    sel1, sel2 = cmd.get_object_list("all")
    # Select only C alphas
    sel1 += ' and name CA'
    sel2 += ' and name CA'
    cutoffs = [1., 2., 4., 8.]
    cmd.align(sel1, sel2, cycles=0, transform=0, object='aln')
    mapping = cmd.get_raw_alignment('aln')  # e.g. [[('prot1', 2530), ('prot2', 2540)], ...]
    distances = []
    for mapping_ in mapping:
        atom1 = '%s and id %d' % (mapping_[0][0], mapping_[0][1])
        atom2 = '%s and id %d' % (mapping_[1][0], mapping_[1][1])
        dist = cmd.get_distance(atom1, atom2)
        cmd.alter(atom1, 'b = %.4f' % dist)
        distances.append(dist)
    distances = np.asarray(distances)
    gdts = []
    for cutoff in cutoffs:
        gdt_cutoff = (distances <= cutoff).sum() / (len(distances))
        gdts.append(gdts)
    out = np.asarray(zip(cutoffs, gdts)).flatten()
    mean_gdt = np.mean(gdts)
    return out, mean_gdt


def calculate_RMSD(pdb_original_path, pdb_predicted_path) -> float:
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    cmd.delete("all")
    cmd.load(pdb_original_path)
    cmd.load(pdb_predicted_path)
    sel1, sel2 = cmd.get_object_list("all")
    # Select only C alphas
    sel1 += " and name CA"
    sel2 += " and name CA"
    cmd.align(sel1, sel2, cycles=0, transform=0, object="aln")
    mapping = cmd.get_raw_alignment(
        "aln"
    )
    rmsd = cmd.align(sel1, sel2)[0]

    return rmsd


def main(args):
    args.af2_results_path = Path(args.af2_results_path)
    args.fasta_path = Path(args.fasta_path)
    args.pdb_path = Path(args.pdb_path)
    assert (
        args.af2_results_path.exists()
    ), f"AF2 file path {args.af2_results_path} does not exist"
    assert args.fasta_path.exists(), f"Fasta file path {args.fasta_path} does not exist"
    assert args.pdb_path.exists(), f"PDB file path {args.pdb_path} does not exist"
    all_fasta_paths = list(args.fasta_path.glob("**/*.fasta"))

    all_results = []
    for fasta_path in all_fasta_paths:
        # Load .fasta:
        with open(fasta_path, "r") as f:
            lines = f.readlines()
            pdb, temp, n = lines[0].strip(">").split("_")
        # Get rid of .fasta extension and extracts path
        # eg: ../data/sampling/fasta/TIMED_1/TIMED_2.fasta -> /TIMED_1/TIMED_2
        root_path = str(fasta_path.with_suffix("")).split(str(args.fasta_path))[1]
        # Find af2 path:
        af2_path = args.af2_results_path / root_path
        assert af2_path.exists(), f"File path {af2_path} does not exist"
        # Find all PDBs in af2 path:
        all_af2_paths = list(args.fasta_path.glob("*.pdb"))
        pdb_path = args.pdb_path / pdb[1:3] / (pdb[:4] + ".pdb1")
        assert pdb_path.exists(), f"PDB path {pdb_path} does not exist"
        curr_results = [
            pdb + f"_{n}", temp
        ]
        for curr_path in all_af2_paths:
            curr_rmsd = calculate_RMSD(curr_path, pdb_path)
            curr_results.append(curr_rmsd)
        all_results.append(curr_results)
    all_results = np.array(all_results)
    np.savetxt("all_results.csv", all_results, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--af2_results_path", type=str, help="Path to input file")
    parser.add_argument("--fasta_path", type=str, help="Path to input file")
    parser.add_argument("--pdb_paths", type=str, help="Path to input file")

    params = parser.parse_args()
    main(params)
