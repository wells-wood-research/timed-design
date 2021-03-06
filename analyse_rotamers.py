import argparse
from pathlib import Path

import numpy as np
from ampal.amino_acids import standard_amino_acids

from design_utils.analyse_utils import (
    analyse_with_scwrl,
    calculate_rotamer_metrics,
    tag_pdb_with_rot,
)
from design_utils.utils import (
    extract_sequence_from_pred_matrix,
    get_rotamer_codec,
    load_datasetmap,
)


def main(args):
    # Sanitise paths:
    args.path_to_pred_matrix = Path(args.path_to_pred_matrix)
    model_name = args.path_to_pred_matrix.stem
    args.output_path = Path(f"{args.output_path}_{model_name}")
    # Create output folder if it does not exist:
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    args.path_to_pdb = Path(args.path_to_pdb)
    # Check paths exist:
    assert (
        args.path_to_pred_matrix.exists()
    ), f"Input file {args.path_to_pred_matrix} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Datasetmap file {args.path_to_datasetmap} does not exist"
    assert args.path_to_pdb.exists(), f"PDB folder {args.path_to_pdb} does not exist"
    # Load datasetmap:
    datasetmap = load_datasetmap(
        args.path_to_datasetmap, is_old=args.support_old_datasetmap
    )
    # Extract PDB codes to be analysed from path:
    pdb_codes = np.unique(datasetmap[:, 0])
    wt_results_dict, pdb_to_assemblies = tag_pdb_with_rot(
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
    # NB: As the new datasetmap format removed the real sequence of residues, this step is necessary to build the pdb_to_real_sequence:
    if not args.support_old_datasetmap:
        for pdb in pdb_to_real_sequence.keys():
            pdb_to_real_sequence[pdb] = pdb_to_assemblies[pdb[:4]][pdb[-1]].sequence
    # Calculate Metrics:
    # - SCWRL_WT
    # - WT
    # - Rotamer
    # - SCWRL_Rotamer
    # Analyses:
    # - #1 WT vs Rotamer: Real rotamer accuracy ie. of the predicted rotamers how many are the same as in the real structure
    # - #2 Rotamer vs SCWRL_Rotamer: Rotamer accuracy from predicted sequence ie. when we predict a rotamer, is it the correct one?
    # - #3 WT vs SCWRL_Rotamer:

    # - Analysis 1: WT_SCWRL vs Rotamer from crystal structure
    calculate_rotamer_metrics(
        pdb_to_probability,
        wt_results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_wt",
        output_path=args.output_path,
    )
    # - Analysis 2: Rotamer vs SCWRL_Rotamer (sequence put through SCWRL)
    #     Analyse rotamers with SCWRL (requires SCWRL install)
    #     First the sequence is packed with SCWRL and saved to PDB,
    #     Then, the same metrics as before are calculated and saved
    pdb_to_scores_rot, _ = analyse_with_scwrl(
        pdb_to_sequence,
        pdb_to_assemblies,
        args.output_path,
        suffix=f"_{model_name}",
        scwrl_path=args.scwrl_path,
    )
    model_pdb_codes = np.core.defchararray.add(pdb_codes, f"_{model_name}")
    rotamer_model_results_dict, _ = tag_pdb_with_rot(
        args.workers, args.output_path, model_pdb_codes
    )
    calculate_rotamer_metrics(
        pdb_to_probability,
        rotamer_model_results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_scwrl_{model_name}",
        output_path=args.output_path,
    )
    # - Analysis 3: TIMED_rotamer vs Real sequence from crystal put through SCWRL
    pdb_to_scores_real, _ = analyse_with_scwrl(
        pdb_to_real_sequence,
        pdb_to_assemblies,
        args.output_path,
        suffix="_scwrl",
        scwrl_path=args.scwrl_path,
    )
    scwrl_pdb_codes = np.core.defchararray.add(pdb_codes, "_scwrl")
    scwrl_results_dict, _ = tag_pdb_with_rot(
        args.workers, args.output_path, scwrl_pdb_codes
    )
    calculate_rotamer_metrics(
        pdb_to_probability,
        scwrl_results_dict,
        flat_categories,
        suffix=f"{model_name}_vs_wt_scwrl",
        output_path=args.output_path,
    )

    # Finally, save all SCWRL Scores to file:
    outfile_scwrl_score = args.output_path / "scwrl_scores.csv"
    with open(outfile_scwrl_score, "w") as f:
        f.write(f"PDB,score_rot,score_real\n")
        for pdb in pdb_to_scores_rot.keys():
            score_rot = pdb_to_scores_rot[pdb]
            score_real = pdb_to_scores_real[pdb]
            f.write(f"{pdb},{score_rot},{score_real}\n")


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
    parser.add_argument(
        "--scwrl_path",
        default="/Users/leo/scwrl4/Scwrl4",
        type=str,
        help="Path to Scwrl4 software",
    )
    params = parser.parse_args()
    main(params)
