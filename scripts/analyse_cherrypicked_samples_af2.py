import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pymol
import seaborn as sns

from design_utils.analyse_utils import extract_packdensity_from_ampal, extract_bfactor_from_ampal, extract_prediction_entropy_to_dict

plt.style.use(["ipynb", "use_mathtext", "colors5-light"])


def plot_packingdensity_vs_position(all_pdbs):
    all_models = []
    all_results = []
    for model_pdb in list(all_pdbs):
        curr_packdensity = extract_packdensity_from_ampal(model_pdb)
        model_name = model_pdb.parent.name
        all_results += curr_packdensity
        all_models.append(model_name)

    all_results = pd.DataFrame(np.array(all_results).T, columns=all_models)
    # TODO Fix:
    fig, axs = plt.subplots(1, figsize=(10, 4.8))
    g = sns.lineplot(data=all_results, ax=axs)
    g.set(ylim=(0, 100))

    axs.yaxis.get_major_formatter().set_scientific(False)
    axs.xaxis.get_major_formatter().set_scientific(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("Packing Density")
    fig.tight_layout()
    plt.show()
    return


def plot_RMSD_vs_position(reference_pdb, all_pdbs):
    all_models = []
    all_results = []
    for model_pdb in list(all_pdbs):
        curr_rmsd, curr_distances = _calculate_RMSD(reference_pdb, model_pdb)
        model_name = model_pdb.parent.name
        all_results += [curr_distances]
        all_models.append(model_name)
    all_results = pd.DataFrame(np.array(all_results).T, columns=all_models)
    # TODO Fix:
    fig, axs = plt.subplots(1, figsize=(10, 4.8))
    sns.lineplot(data=all_results, ax=axs)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    plt.show()

    return


def plot_af2IDDT_vs_position(all_pdbs):
    all_models = []
    all_results = []
    for model_pdb in list(all_pdbs):
        curr_b_factors = extract_bfactor_from_ampal(model_pdb)
        model_name = model_pdb.parent.name
        all_results += curr_b_factors
        all_models.append(model_name)

    all_results = pd.DataFrame(np.array(all_results).T, columns=all_models)
    # TODO Fix:
    fig, axs = plt.subplots(1, figsize=(10, 4.8))
    g = sns.lineplot(data=all_results, ax=axs)
    g.set(ylim=(0, 100))

    axs.yaxis.get_major_formatter().set_scientific(False)
    axs.xaxis.get_major_formatter().set_scientific(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    plt.show()
    return


def plot_entropy_vs_position(timed_pred_folder, models, pdb_code):
    all_results = []
    all_models = []
    for model in models:
        model_pred_path = timed_pred_folder / (model + ".csv")
        model_map_path = timed_pred_folder / (model + ".txt")
        pdb_to_entropy = extract_prediction_entropy_to_dict(model_pred_path, model_map_path, rotamer_mode=True if "rot" in model else False)
        pdb_code_and_chain = 0
        # Go through keys and identify the correct chain:
        for pc in pdb_to_entropy.keys():
            if pdb_code in pc:
                pdb_code_and_chain = pc
                # End loop once found:
                break
        assert pdb_code_and_chain != 0, f"PDB Code {pdb_code} not found in {list(pdb_to_entropy.keys())}"
        curr_entropy = pdb_to_entropy[pdb_code_and_chain]
        all_results.append(curr_entropy)
        all_models.append(model)

    all_results = pd.DataFrame(np.array(all_results).T, columns=all_models)
    # TODO Fix:
    fig, axs = plt.subplots(1, figsize=(10, 4.8))
    g = sns.lineplot(data=all_results, ax=axs)
    g.set(ylim=(0, 5))

    axs.yaxis.get_major_formatter().set_scientific(False)
    axs.xaxis.get_major_formatter().set_scientific(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    plt.show()
    return


def _calculate_RMSD(pdb_original_path, pdb_predicted_path) -> (float, float):
    # pymol.pymol_argv = ["pymol", "-qc"]
    # pymol.finish_launching()
    # cmd = pymol.cmd
    # cmd.undo_disable()  # Avoids pymol giving errors with memory
    # cmd.delete("all")
    # cmd.load(pdb_original_path, object="refori")
    # cmd.load(pdb_predicted_path, object="modelled")
    # sel_ref, sel_model = cmd.get_object_list("all")
    # # Select only C alphas
    # sel_ref += " and name CA"
    # sel_model += " and name CA"
    # rmsd = cmd.cealign(target=sel_ref, mobile=sel_model)['RMSD']
    # cmd.cealign(target=sel_ref, mobile=sel_model, transform=0, object="aln")
    # mapping = cmd.get_raw_alignment("aln")
    # distances = []
    # for mapping_ in mapping:
    #     try:
    #         atom1 = f"{mapping_[0][0]} and id {mapping_[0][1]}"
    #         atom2 = f"{mapping_[1][0]} and id {mapping_[1][1]}"
    #         dist = cmd.get_distance(atom1, atom2)
    #         cmd.alter(atom1, f"b = {dist:.4f}")
    #         distances.append(dist)
    #     except:
    #         continue
    # distances = np.asarray(distances)
    # # TODO: Deal with distances in a better way
    # return rmsd, distances
    return 3, np.random.randint(4, size=10)


def main(args):
    args.reference_pdb = Path(args.reference_pdb)
    args.af2_model_pdb = Path(args.af2_model_pdb)
    args.timed_pred_folder = Path(args.timed_pred_folder)
    assert (
        args.reference_pdb.exists()
    ), f"Input file {args.reference_pdb} does not exist"
    assert (
        args.af2_model_pdb.exists()
    ), f"Input file {args.af2_model_pdb} does not exist"
    # Find all PDBs matching the reference PDB code
    pdb_code = args.reference_pdb.name.split(".")[0]
    all_pdbs = list(args.af2_model_pdb.glob(f"**/*{pdb_code}*_ranked_0.pdb"))
    # Plot RMSD:
    plot_RMSD_vs_position(args.reference_pdb, all_pdbs)
    plot_af2IDDT_vs_position(all_pdbs)
    plot_packingdensity_vs_position(all_pdbs)
    # Extract all models:
    all_models = [model_pdb.parent.name for model_pdb in all_pdbs]
    plot_entropy_vs_position(args.timed_pred_folder, all_models, pdb_code)
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--reference_pdb", type=str, help="Path to reference file")
    parser.add_argument("--af2_model_pdb", type=str, help="Path to alphafold file")
    parser.add_argument("--timed_pred_folder", type=str, help="Path to folder with predictions per model")
    params = parser.parse_args()
    main(params)
