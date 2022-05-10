import argparse
from pathlib import Path

import numpy as np
import pymol


def calculate_GDT(pdb_original_path, pdb_predicted_path):
    """
    sel1: protein to compare
    sel2: reference protein
    """
    cmd.delete("all")
    cmd.load(pdb_original_path)
    cmd.load(pdb_predicted_path)
    sel1, sel2 = cmd.get_object_list("all")
    # Select only C alphas
    sel1 += " and name CA"
    sel2 += " and name CA"
    cutoffs = [1.0, 2.0, 4.0, 8.0]
    cmd.align(sel1, sel2, cycles=0, transform=0, object="aln")
    mapping = cmd.get_raw_alignment(
        "aln"
    )  # e.g. [[('prot1', 2530), ('prot2', 2540)], ...]
    distances = []
    for mapping_ in mapping:
        atom1 = "%s and id %d" % (mapping_[0][0], mapping_[0][1])
        atom2 = "%s and id %d" % (mapping_[1][0], mapping_[1][1])
        dist = cmd.get_distance(atom1, atom2)
        cmd.alter(atom1, "b = %.4f" % dist)
        distances.append(dist)
    distances = np.asarray(distances)
    gdts = []
    for cutoff in cutoffs:
        gdt_cutoff = (distances <= cutoff).sum() / (len(distances))
        gdts.append(gdt_cutoff)
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
    mapping = cmd.get_raw_alignment("aln")
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
    error_log = []
    all_fasta_paths = list(args.fasta_path.glob("**/*.fasta"))
    all_results = []
    for fasta_path in all_fasta_paths:
        # Load .fasta:
        with open(fasta_path, "r") as f:
            lines = f.readlines()
            model, pdb, temp, n = lines[0].strip(">").strip("\n").rsplit("_", 3)
        # Get rid of .fasta extension and extracts path
        # eg: ../data/sampling/fasta/TIMED_1/TIMED_2.fasta -> TIMED_1/TIMED_2
        # The last [1:] gets rid of the starting "/" character
        root_path = str(fasta_path.with_suffix("")).split(str(args.fasta_path))[1][1:]
        # Find af2 path:
        af2_path = args.af2_results_path / root_path
        assert af2_path.exists(), f"File path {af2_path} does not exist"
        # Find all PDBs in af2 path:
        all_af2_paths = list(af2_path.glob("*.pdb"))
        pdb_path = args.pdb_path / (pdb[:4] + ".pdb")
        assert pdb_path.exists(), f"PDB path {pdb_path} does not exist"
        curr_results = [model, pdb, n, temp]
        # TODO: Add multiprocessing?
        for curr_path in all_af2_paths:
            try:
                curr_rmsd = calculate_RMSD(curr_path, pdb_path)
            except:
                curr_rmsd = 0
                error_log.append((pdb_path, str(curr_path)) )
            curr_results.append(curr_rmsd)
        all_results.append(curr_results)
    all_results = np.array(all_results)
    np.savetxt("all_results.csv", all_results, delimiter=",", fmt='%s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--af2_results_path", type=str, help="Path to input file")
    parser.add_argument("--fasta_path", type=str, help="Path to input file")
    parser.add_argument("--pdb_path", type=str, help="Path to input file")
    # Launch PyMol:
    params = parser.parse_args()
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    main(params)
