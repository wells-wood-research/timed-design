import argparse
import tempfile
from pathlib import Path

import ampal
import numpy as np
import pymol
from tqdm import tqdm


def calculate_RMSD_and_gdt(pdb_original_path, pdb_predicted_path) -> float:
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    cmd.delete("all")
    cmd.load(pdb_original_path, object="reference")
    cmd.load(pdb_predicted_path, object="model")
    sel_ref, sel_model = cmd.get_object_list("all")
    # Select only C alphas
    sel_ref += " and name CA"
    sel_model += " and name CA"
    rmsd = cmd.align(target=sel_ref, mobile=sel_model)[0]
    cmd.align(target=sel_ref, mobile=sel_model, cycles=0, transform=0, object="aln")
    mapping = cmd.get_raw_alignment("aln")

    cutoffs = [1.0, 2.0, 4.0, 8.0]
    distances = []
    for mapping_ in mapping:
        atom1 = f"{mapping_[0][0]} and id {mapping_[0][1]}"
        atom2 = f"{mapping_[1][0]} and id {mapping_[1][1]}"
        dist = cmd.get_distance(atom1, atom2)
        cmd.alter(atom1, f"b = {dist:.4f}")
        distances.append(dist)
    distances = np.asarray(distances)
    gdts = []
    for cutoff in cutoffs:
        gdt_cutoff = (distances <= cutoff).sum() / (len(distances))
        gdts.append(gdt_cutoff)

    mean_gdt = np.mean(gdts)
    return rmsd, mean_gdt


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
    for fasta_path in tqdm(all_fasta_paths, desc="Analysing Fasta"):
        # Load .fasta:
        with open(fasta_path, "r") as f:
            lines = f.readlines()
            model, pdb, temp, n = lines[0].strip(">").strip("\n").rsplit("_", 3)
            seq = lines[1].strip("\n")
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
            curr_pdb = ampal.load_pdb(str(curr_path))
            assert (
                curr_pdb.sequences[0] == seq
            ), f"Sequence {fasta_path} at {lines[0]} and curr_pdb {curr_path} do not match."
            reference_pdb = ampal.load_pdb(str(pdb_path))
            assert (
                reference_pdb.sequences[0] == seq
            ), f"Reference Sequence {pdb_path} and sequence {fasta_path} do not match."
            assert len(curr_pdb.sequences[0]) == len(
                reference_pdb.sequences[0]
            ), f"Length of reference sequence and current pdb do not match"

            with tempfile.NamedTemporaryFile() as reference_pdb_tmp_path, tempfile.NamedTemporaryFile() as curr_pdb_tmp_path:
                # Pre-process with ampal to avoid junk:
                reference_pdb_tmp_path.write(curr_pdb.pdb)
                reference_pdb_tmp_path.seek(0)
                # Resets the buffer back to the first line
                curr_pdb_tmp_path.write(reference_pdb.pdb)
                curr_pdb_tmp_path.seek(0)
                curr_rmsd, curr_gdt = calculate_RMSD_and_gdt(
                    reference_pdb_tmp_path, curr_pdb_tmp_path
                )
                # curr_rmsd = np.nan
                # error_log.append((pdb_path, str(curr_path)))
                curr_results.append(curr_rmsd)
                curr_results.append(curr_gdt)
        all_results.append(curr_results)
    all_results = np.array(all_results)
    np.savetxt("all_results.csv", all_results, delimiter=",", fmt="%s")
    np.savetxt("errors.csv", np.array(error_log), delimiter=",", fmt="%s")


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
