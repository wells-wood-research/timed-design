"""
Similar to analyse_af2.py but for output of move_af2_pdb
"""
import argparse
import tempfile
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path

import ampal
import numpy as np
import pymol
from sklearn import metrics


def calculate_RMSD_and_gdt(pdb_original_path, pdb_predicted_path) -> (float, float):
    pymol.pymol_argv = ["pymol", "-qc"]
    pymol.finish_launching()
    cmd = pymol.cmd
    cmd.undo_disable()  # Avoids pymol giving errors with memory
    cmd.delete("all")
    cmd.load(pdb_original_path, object="refori")
    cmd.load(pdb_predicted_path, object="modelled")
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
        try:
            atom1 = f"{mapping_[0][0]} and id {mapping_[0][1]}"
            atom2 = f"{mapping_[1][0]} and id {mapping_[1][1]}"
            dist = cmd.get_distance(atom1, atom2)
            cmd.alter(atom1, f"b = {dist:.4f}")
            distances.append(dist)
        except:
            continue
    distances = np.asarray(distances)
    gdts = []
    for cutoff in cutoffs:
        gdt_cutoff = (distances <= cutoff).sum() / (len(distances))
        gdts.append(gdt_cutoff)

    mean_gdt = np.mean(gdts)
    return rmsd, mean_gdt

def analyse_pdb_path(curr_path, args):
    model, pdb, temp, n, af2_model = curr_path.name.split("_", maxsplit=4)
    # Load af2 pdb structures to sanitise input:
    curr_pdb = ampal.load_pdb(str(curr_path))
    if isinstance(curr_pdb, ampal.AmpalContainer):
        curr_pdb = curr_pdb[0]
    pdb_path = args.pdb_path / pdb[1:3] / (pdb[:4] + ".pdb1")
    # Load reference pdb structures to sanitise input:
    reference_pdb = ampal.load_pdb(str(pdb_path))
    if isinstance(reference_pdb, ampal.AmpalContainer):
        reference_pdb = reference_pdb[0]
    try:
        # Sanity checks:
        assert len(curr_pdb.sequences[0]) == len(
            reference_pdb.sequences[0]
        ), f"Length of reference sequence and current pdb do not match for {pdb}: {len(curr_pdb.sequences[0])} vs {len(reference_pdb.sequences[0])}"
    except AssertionError:
        return None
    # Calculate accuracy:
    seq_accuracy = metrics.accuracy_score(list(curr_pdb.sequences[0]), list(reference_pdb.sequences[0]))
    curr_results = [model, pdb, n, temp, seq_accuracy]
    # Required purely to avoid bugs on files being corrupted:
    with tempfile.NamedTemporaryFile(
            mode="w",
            delete=True,
            suffix=".pdb",
    ) as reference_pdb_tmp_path, tempfile.NamedTemporaryFile(
        mode="w",
        delete=True,
        suffix=".pdb",
    ) as curr_pdb_tmp_path:
        # Pre-process with ampal to avoid junk:
        reference_pdb_tmp_path.write(curr_pdb.pdb)
        reference_pdb_tmp_path.seek(0)
        # Resets the buffer back to the first line
        curr_pdb_tmp_path.write(reference_pdb.pdb)
        curr_pdb_tmp_path.seek(0)
        # Calculate metrics:
        curr_rmsd, curr_gdt = calculate_RMSD_and_gdt(
            reference_pdb_tmp_path.name, curr_pdb_tmp_path.name
        )
        # Append current metrics
        curr_results.append(curr_rmsd)
        curr_results.append(curr_gdt)
    # Pool all metrics together:
    return curr_results

def main(args):
    args.af2_results_path = Path(args.af2_results_path)
    args.pdb_path = Path(args.pdb_path)
    assert (
        args.af2_results_path.exists()
    ), f"AF2 file path {args.af2_results_path} does not exist"
    assert args.pdb_path.exists(), f"PDB file path {args.pdb_path} does not exist"
    error_log = []
    # Find all PDBs in af2 path:
    all_af2_paths = list(args.af2_results_path.glob("*_ranked_*.pdb"))
    with Pool(processes=args.workers) as p:
        all_results = p.starmap(
            analyse_pdb_path,
            zip(
                all_af2_paths,
                repeat(args),
            ),
        )
        p.close()
    # Load results and save as csv:
    all_results = np.array(all_results)
    np.savetxt(f"all_results_{all_results[0][0]}.csv", all_results, delimiter=",", fmt="%s")
    np.savetxt(f"errors_{all_results[0][0]}.csv", np.array(error_log), delimiter=",", fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--af2_results_path", type=str, help="Path to input file")
    parser.add_argument("--pdb_path", type=str, help="Path to input file")
    parser.add_argument("--workers", type=int, help="Path to input file")
    # Launch PyMol:
    params = parser.parse_args()
    main(params)
