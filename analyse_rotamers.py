import argparse
from pathlib import Path

import numpy as np
from ampal.amino_acids import standard_amino_acids

from utils.analyse_utils import analyse_with_scwrl, calculate_metrics, \
    tag_pdb_with_rot
from utils.utils import (
    extract_sequence_from_pred_matrix,
    get_rotamer_codec,
    load_datasetmap,
)


def main(args):
    args.path_to_pred_matrix = Path(args.path_to_pred_matrix)
    model_name = args.path_to_pred_matrix.stem
    args.output_path = Path(f"{args.output_path}_{model_name}")
    # Create output folder if it does not exist:
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    args.path_to_pdb = Path(args.path_to_pdb)
    assert (
        args.path_to_pred_matrix.exists()
    ), f"Input file {args.path_to_pred_matrix} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Datasetmap file {args.path_to_datasetmap} does not exist"
    assert args.path_to_pdb.exists(), f"PDB folder {args.path_to_pdb} does not exist"
    # Load datasetmap
    datasetmap = load_datasetmap(
        args.path_to_datasetmap, is_old=args.support_old_datasetmap
    )
    # Extract PDB codes
    pdb_codes = np.unique(datasetmap[:, 0])
    results_dict, pdb_to_assemblies = tag_pdb_with_rot(
        args.workers, args.path_to_pdb, pdb_codes
    )
    # Load prediction matrix of model of interest:
    prediction_matrix = np.genfromtxt(
        args.path_to_pred_matrix, delimiter=",", dtype=np.float16
    )
    # Get rotamer categories:
    _, flat_categories = get_rotamer_codec()
    # Get dictionary for 3 letter -> 1 letter conversion:
    res_to_r = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
    # Create flat categories of 1 letter amino acid for each of the 338 rotamers:
    rotamers_categories = [res_to_r[res.split("_")[0]] for res in flat_categories]
    # Extract dictionaries with sequences:
    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    ) = extract_sequence_from_pred_matrix(
        datasetmap,
        prediction_matrix,
        rotamers_categories=rotamers_categories,
        old_datasetmap=args.support_old_datasetmap,
    )
    # Calculate Metrics:
    # - Analysis 1: TIMED_rotamer vs real rotamers from crystal structure
    calculate_metrics(
        pdb_to_probability,
        results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_original",
    )

    # - Analysis 2: TIMED_rotamer vs TIMED_rotamer sequence put through SCWRL
    #     Analyse rotamers with SCWRL (requires SCWRL install)
    #     First the sequence is packed with SCWRL and saved to PDB,
    #     Then, the same metrics as before are calculated and saved
    pdb_to_sequence = {pdb_codes[0]: pdb_to_sequence[pdb_codes[0]]}
    pdb_to_assemblies = {pdb_codes[0]: pdb_to_assemblies[pdb_codes[0]]}
    # Analyse with SCWRL and get scores
    pdb_to_scores_rot = analyse_with_scwrl(
        pdb_to_sequence, pdb_to_assemblies, args.output_path, suffix=model_name
    )
    model_pdb_codes = np.core.defchararray.add(pdb_codes, model_name)
    model_results_dict, _ = tag_pdb_with_rot(
        args.workers, args.output_path, model_pdb_codes
    )
    calculate_metrics(
        pdb_to_probability,
        model_results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_pred+scwrl",
    )
    # - Analysis 3: TIMED_rotamer vs Real sequence from crystal put through SCWRL
    pdb_to_scores_real = analyse_with_scwrl(
        pdb_to_real_sequence, pdb_to_assemblies, args.output_path, suffix="scwrl"
    )
    scwrl_pdb_codes = np.core.defchararray.add(pdb_codes, ("_" + "scwrl"))
    scwrl_results_dict, _ = tag_pdb_with_rot(
        args.workers, args.path_to_pdb, scwrl_pdb_codes
    )
    calculate_metrics(
        pdb_to_probability,
        scwrl_results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_ori+scwrl",
    )
    # Save SCWRL Scores
    outfile_scwrl_score = args.output_path / "scwrl_scores.csv"
    with open(outfile_scwrl_score, "w") as f:
        f.write(f"PDB,score_rot,score_real")
        for pdb in pdb_to_scores_rot.keys():
            score_rot = pdb_to_scores_rot[pdb]
            score_real = pdb_to_scores_real[pdb]
            f.write(f"{pdb},{score_rot},{score_real}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--path_to_pred_matrix", type=str, help="Path to model .csv file"
    )
    parser.add_argument(
        "--output_path", default="output", type=str, help="Path to save analysis"
    )
    parser.add_argument(
        "--path_to_pdb",
        type=str,
        help="Path to biounit pdb dataset. Needs to be in format pdb/{2nd and 3rd char}/{pdb}.pdb1.gz",
    )
    parser.add_argument(
        "--path_to_datasetmap",
        default="datasetmap.txt",
        type=str,
        help="Path to dataset map ending with .txt",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    parser.add_argument(
        "--support_old_datasetmap",
        default=False,
        action="store_true",
        help="Whether model to import from the old datasetmap (default: False)",
    )
    params = parser.parse_args()
    main(params)
