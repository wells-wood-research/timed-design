import argparse
from collections import Counter
import gzip
import typing as t
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path


import ampal
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, top_k_accuracy_score, precision_score, recall_score
from ampal.amino_acids import standard_amino_acids
from aposteriori.data_prep.create_frame_data_set import _fetch_pdb
from utils import get_rotamer_codec, extract_sequence_from_pred_matrix


def calulate_metrics(pdb_to_probability, pdb_to_rotamer, rot_categories):
    y_pred = []
    y_true = []
    # Extract predictions:
    for pdb in pdb_to_rotamer.keys():
        y_pred.append(pdb_to_probability[pdb])
        y_true.append(pdb_to_rotamer[pdb])
    y_pred = np.array(y_pred).reshape(-1, 338)
    y_true = np.array(y_true).flatten()
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    # Sometimes predictions miss a very small amount due to the way they are saved:
    if not np.allclose(1, y_pred.sum(axis=1)):
        idx_res = y_pred.sum(axis=1) != 1
        old_residuals = y_pred[idx_res].sum(axis=1)
        # Calculate residual and then create idx_res, 338 with the residual / 338
        residual = np.ones((338, idx_res.sum())) * (1-y_pred[idx_res].sum(axis=1)) / 338
        y_pred[idx_res] = np.add(y_pred[idx_res], residual.T)
        assert np.allclose(1, y_pred.sum(axis=1)), f"Probabilities at idx {idx_res} do not add up to 1: {old_residuals} and after adjustment got {y_pred[idx_res]}"
    y_argmax = np.argmax(y_pred, axis=1)
    # Calculate metrics:
    auc_ovo = roc_auc_score(y_true, y_pred, multi_class="ovo", labels=list(range(len(rot_categories))), average='macro')
    try:
        auc_ovr = roc_auc_score(y_true, y_pred, multi_class="ovr", labels=list(range(len(rot_categories))), average='macro')
    except ValueError:
        auc_ovr = np.nan
    report = classification_report(y_true, y_argmax, labels=list(range(len(rot_categories))))
    accuracy = accuracy_score(y_true, y_argmax)
    accuracy_2 = top_k_accuracy_score(y_true, y_pred, k=2, labels=list(range(len(rot_categories))))
    accuracy_3 = top_k_accuracy_score(y_true, y_pred, k=3, labels=list(range(len(rot_categories))))
    accuracy_4 = top_k_accuracy_score(y_true, y_pred, k=4, labels=list(range(len(rot_categories))))
    accuracy_5 = top_k_accuracy_score(y_true, y_pred, k=5, labels=list(range(len(rot_categories))))
    precision = precision_score(y_true, y_argmax, average="macro", labels=list(range(len(rot_categories))), zero_division=0)
    recall = recall_score(y_true, y_argmax, average="macro", labels=list(range(len(rot_categories))), zero_division=0)
    print("Metrics AUC_OVR")
    print(auc_ovr)
    print("Metrics AUC_OVO")
    print(auc_ovo)
    print("Metrics Report")
    print(report)
    print(f"report: {report}, accuracy: {accuracy}, accuracy_2: {accuracy_2}, accuracy_3: {accuracy_3}, accuracy_4: {accuracy_4}, accuracy_5: {accuracy_5}, precision: {precision}, recall: {recall}" )
    # Calculate bias:
    count_labels = Counter(y_true)
    count_pred = Counter(y_argmax)
    bias = {}
    sum_counts = len(y_true)
    for y in count_labels.keys():
        if y in count_labels and y in count_pred:
            c_label = count_labels[int(y)] / sum_counts
            c_pred = count_pred[int(y)] / sum_counts
            b = c_pred - c_label
            bias[rot_categories[int(y)]] = b
        else:
            bias[rot_categories[int(y)]] = np.nan
    print(count_labels)
    print(count_pred)
    print(bias)


def extract_rotamer_encoding(pdb_code, monomer):
    _, flat_categories = get_rotamer_codec()
    res_rot_to_encoding = dict(zip(flat_categories, range(len(flat_categories))))
    all_rot = []
    for res in monomer:
        try:
            rot_key = f"{res.mol_code}_{''.join(map(str, res.tags['rotamers']))}"
            if rot_key in res_rot_to_encoding:
                all_rot.append(res_rot_to_encoding[rot_key])
            else:
                all_rot.append(np.nan)
        except TypeError:
            all_rot.append(np.nan)

    return {f"{pdb_code}{monomer.id}": all_rot}


def tag_pdb_with_rot(pdb_code, pdb_path):
    # pdb_path = pdb_path / pdb_code[1:3] / (pdb_code + ".pdb1.gz")
    out_dir =  pdb_path / pdb_code[1:3]
    pdb_path = out_dir / (pdb_code + ".pdb1.gz")
    if not pdb_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = out_dir / (pdb_code + ".pdb1")
        if not pdb_path.exists():
            pdb_path = _fetch_pdb(pdb_code, verbosity=1, output_folder=out_dir)
        assembly = ampal.load_pdb(pdb_path)
    else:
        with gzip.open(str(pdb_path), "rb") as inf:
            assembly = ampal.load_pdb(inf.read().decode(), path=False)
    if isinstance(assembly, ampal.AmpalContainer):
        assembly = assembly[0]
    if isinstance(assembly, ampal.Assembly):
        # For each monomer in the assembly:
        result_dict = {}
        for monomer in assembly:
            if isinstance(monomer, ampal.Polypeptide):
                monomer.tag_sidechain_dihedrals()
                curr_result_dict = extract_rotamer_encoding(pdb_code, monomer)
                result_dict.update(curr_result_dict)
    elif isinstance(assembly, ampal.Polypeptide):
        assembly.tag_sidechain_dihedrals()
        result_dict = extract_rotamer_encoding(pdb_code, assembly)

    return result_dict


def main(args):
    args.input_path = Path(args.input_path)
    args.path_to_datasetmap = Path(args.path_to_datasetmap)
    args.path_to_pdb = Path(args.path_to_pdb)
    assert args.input_path.exists(), f"Input file {args.input_path} does not exist"
    assert (
        args.path_to_datasetmap.exists()
    ), f"Datasetmap file {args.path_to_datasetmap} does not exist"
    assert args.path_to_pdb.exists(), f"PDB folder {args.path_to_pdb} does not exist"
    datasetmap = np.genfromtxt(f"{args.path_to_datasetmap}", delimiter=",", dtype=str)
    pdb_codes = np.unique(datasetmap[:, 0])
    results_dict = {}
    with Pool(processes=args.workers) as p:
        results_dict_list = p.starmap(
            tag_pdb_with_rot,
            zip(
                pdb_codes,
                repeat(args.path_to_pdb),
            ),
        )
        p.close()
    for curr_res_dict in results_dict_list:
        results_dict.update(curr_res_dict)
    prediction_matrix = np.genfromtxt(args.input_path, delimiter=",", dtype=np.float16)
    _, flat_categories = get_rotamer_codec()
    # TODO keep these categories for later
    res_to_r = dict(zip(standard_amino_acids.values(), standard_amino_acids.keys()))
    # Create flat categories of 1 letter amino acid for each of the 338 rotamers
    res_categories = [res_to_r[res.split("_")[0]] for res in flat_categories]
    (
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    ) = extract_sequence_from_pred_matrix(
        datasetmap, prediction_matrix, codec=True, flat_categories=res_categories
    )
    calulate_metrics(pdb_to_probability, curr_res_dict, flat_categories)
    raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path", type=str, help="Path to model .csv file")
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
    params = parser.parse_args()
    main(params)
