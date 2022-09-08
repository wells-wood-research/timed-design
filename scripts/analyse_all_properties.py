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
from design_utils.analyse_utils import (
    extract_packdensity_from_ampal,
    extract_bfactor_from_ampal,
    extract_prediction_entropy_to_dict,
)


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
    rmsd = cmd.cealign(target=sel_ref, mobile=sel_model)["RMSD"]
    return rmsd


def analyse_pdb_path(curr_path, args, pdb_to_entropy):
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
        return [model, pdb, n, temp, np.nan, np.nan, np.nan]
    # Calculate accuracy:
    seq_accuracy = metrics.accuracy_score(
        list(curr_pdb.sequences[0]), list(reference_pdb.sequences[0])
    )
    # Calculate Packing Density:
    curr_packdensity = extract_packdensity_from_ampal(curr_pdb)
    real_packdensity = extract_packdensity_from_ampal(reference_pdb)
    # Extract AF2 IDDT:
    curr_bfactor = extract_bfactor_from_ampal(curr_pdb)
    curr_entropy = pdb_to_entropy[pdb[:4]]
    # Add to results
    curr_results = [
        model,
        pdb,
        n,
        temp,
        seq_accuracy,
        np.mean(curr_entropy),
        np.std(curr_entropy),
        np.mean(curr_packdensity),
        np.std(curr_packdensity),
        np.mean(real_packdensity),
        np.std(real_packdensity),
        np.mean(curr_bfactor),
        np.std(curr_bfactor),
    ]
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
        curr_rmsd = calculate_RMSD_and_gdt(
            reference_pdb_tmp_path.name, curr_pdb_tmp_path.name
        )
        # Append current metrics
        curr_results.append(curr_rmsd)
    # Pool all metrics together:
    return curr_results


def main(args):
    args.af2_results_path = Path(args.af2_results_path)
    args.pdb_path = Path(args.pdb_path)
    args.timed_pred_folder = Path(args.timed_pred_folder)

    model = args.af2_results_path.name
    model_pred_path = args.timed_pred_folder / (model + ".csv")
    model_map_path = args.timed_pred_folder / (model + ".txt")
    assert model_map_path.exists(), f"File not found {model_map_path}"
    assert model_pred_path.exists(), f"File not found {model_pred_path}"
    pdb_to_entropy = extract_prediction_entropy_to_dict(
        model_pred_path, model_map_path, rotamer_mode=True if "rot" in model else False
    )
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
                repeat(pdb_to_entropy),
            ),
        )
        p.close()
    # Load results and save as csv:
    all_results = np.array(all_results)
    np.savetxt(
        f"all_results_{all_results[0][0]}.csv", all_results, delimiter=",", fmt="%s"
    )
    np.savetxt(
        f"errors_{all_results[0][0]}.csv", np.array(error_log), delimiter=",", fmt="%s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--af2_results_path", type=str, help="Path to input file")
    parser.add_argument("--pdb_path", type=str, help="Path to input file")
    parser.add_argument(
        "--timed_pred_folder",
        type=str,
        help="Path to folder with predictions per model",
    )
    parser.add_argument("--workers", type=int, help="Path to input file")
    # Launch PyMol:
    params = parser.parse_args()
    main(params)
