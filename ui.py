import argparse
import time
import typing as t
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
from ampal.amino_acids import standard_amino_acids
from millify import millify
from sklearn.metrics import accuracy_score
from stmol import showmol

from aposteriori.data_prep.create_frame_data_set import Codec, make_frame_dataset
from sample import main_sample
from design_utils.analyse_utils import (
    calculate_metrics,
    calculate_seq_metrics,
    encode_sequence_to_onehot,
)
from design_utils.utils import get_rotamer_codec, lookup_blosum62
from predict import load_dataset_and_predict


# {{{ Cached Wrappers
@st.cache(show_spinner=False)
def _calculate_seq_metrics_wrapper(seq: str):
    return calculate_seq_metrics(seq)


@st.cache(show_spinner=False)
def _calculate_metrics_wrapper(pdb_to_sequence: dict, pdb_to_real_sequence: dict):
    return calculate_metrics(pdb_to_sequence, pdb_to_real_sequence)


@st.cache(show_spinner=False)
def _calculate_sequence_similarity_wrapper(real_seq: str, predicted_seq: str):
    similarity_score = [
        1 if lookup_blosum62(a, b) > 0 else 0 for a, b in zip(real_seq, predicted_seq)
    ]
    return np.mean(similarity_score)


@st.cache(show_spinner=False)
def _build_aposteriori_dataset_wrapper(
    path_to_pdb: Path, pdb_code: str, output_path: Path, workers: int
):
    structure_path = path_to_pdb / pdb_code[1:3] / (pdb_code + ".pdb1.gz")
    data_path = output_path / (pdb_code + ".hdf5")
    if data_path.exists():
        return data_path
    else:
        make_frame_dataset(
            structure_files=[structure_path],
            output_folder=output_path,
            name=pdb_code,
            frame_edge_length=21.0,
            voxels_per_side=21,
            codec=Codec.CNOCBCA(),
            processes=workers,
            is_pdb_gzipped=True,
            require_confirmation=False,
            voxels_as_gaussian=True,
            voxelise_all_states=False,
        )
    return data_path


@st.cache(show_spinner=False)
def _get_rotamer_codec_wrapper():
    return get_rotamer_codec()


# }}}

# {{{ Utility Functions
@st.cache(show_spinner=False)
def _create_altair_labels(flat_categories) -> str:
    """
    Creates altair-compatible labels

    Parameters
    ----------
    flat_categories: t.List[str]
        List of categories

    Returns
    -------
    altair_labels: str
        Altair-style regex for label conversion
    """
    altair_labels = ""
    for i, cat in enumerate(flat_categories):
        if i == 0:
            base = "datum.label == "
        else:
            base = ":datum.label == "
        if i == len(flat_categories) - 1:
            full = f": '{cat}'"
        else:
            full = base + str(i) + f" ? '{cat}'\n"
        altair_labels += full
    return altair_labels


@st.cache(show_spinner=False)
def predict_dataset(
    dataset_path: str, path_to_model: str, rotamer_mode: bool, model_name_suffix: str
):
    """

    Parameters
    ----------
    dataset_path: str
        Path to dataset
    path_to_model: str
        Path to model
    rotamer_mode: Bool
        Whether to predict rotamer (true) or not (false)
    model_name_suffix: str
        Extra string to append at the end of the files
    """
    with st.spinner("Calculating results.."):
        (
            flat_dataset_map,
            pdb_to_sequence,
            pdb_to_probability,
            pdb_to_real_sequence,
            pdb_to_consensus,
            pdb_to_consensus_prob,
        ) = load_dataset_and_predict(
            [path_to_model],
            Path(dataset_path),
            batch_size=500,
            start_batch=0,
            blacklist=None,
            dataset_map_path=" ",  # Needs to be empty to force creation of new datasetmap
            predict_rotamers=rotamer_mode,
            model_name_suffix=model_name_suffix,
        )
    return (
        flat_dataset_map,
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    )


@st.cache(
    show_spinner=False, allow_output_mutation=True
)  # Output mutation necessary as object changes as it is interacted with
def show_pdb(pdb_code, label_res: t.Optional[str] = None):
    xyzview = py3Dmol.view(query="pdb:" + pdb_code)
    xyzview.setStyle({"cartoon": {"color": "spectrum"}})
    xyzview.setBackgroundColor("#FFFFFF")
    # loop_resid_dict = {sw1_name: sw1_resids, sw2_name: sw2_resids}
    if label_res:
        xyzview.setStyle({"cartoon": {"color": "white", "opacity": 0.5}})
        _, resn, _, chain, _ = label_res.split(" ")
        resn = int(resn)
        zoom_residue = [
            {"resi": int(resn)},
            {
                "backgroundColor": "lightgray",
                "fontColor": "black",
                "backgroundOpacity": 0.5,
            },
            {"stick": {"colorscheme": "default", "radius": 0.2}},
        ]
        xyzview.addResLabels(zoom_residue[0], zoom_residue[1])
        xyzview.addStyle(zoom_residue[0], zoom_residue[2])
        xyzview.addStyle(
            {"resi": f"{(resn-5)}-{resn+5}"},
            {"cartoon": {"color": "orange", "opacity": 0.75}},
        )
        xyzview.zoomTo(zoom_residue[0])
    else:
        xyzview.spin(True)

    return xyzview


@st.cache(show_spinner=False)
def _search_all_pdbs(path_to_pdb: Path):
    all_structures = path_to_pdb.glob(f"**/*.pdb1.gz")
    all_pdbs = [p.stem[:4] for p in all_structures]
    return all_pdbs


@st.cache(show_spinner=False)
def _encode_sequence_to_onehot(pdb_to_sequence: dict, pdb_to_real_sequence: dict):
    return encode_sequence_to_onehot(pdb_to_sequence, pdb_to_real_sequence)


@st.cache(show_spinner=True, allow_output_mutation=True)
def _optimize_seq_with_montecarlo(
    path_to_pred_matrix,
    path_to_datasetmap,
    rotamer_mode,
    sample_n,
    workers,
    temperature,
):
    arguments = [
        "--path_to_pred_matrix",
        path_to_pred_matrix,
        "--path_to_datasetmap",
        path_to_datasetmap,
        "--sample_n",
        str(sample_n),
        "--workers",
        str(workers),
        "--temperature",
        str(temperature),
    ]
    # arguments += [f"--predict_rotamers", True]
    # TODO: There must be a better way to do this:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path_to_pred_matrix")
    parser.add_argument("--path_to_datasetmap")
    parser.add_argument("--sample_n", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--temperature", type=float)
    # Other arguments:
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--support_old_datasetmap", type=int, default=False)
    parser.add_argument(
        "--save_as",
        type=str,
        default="all",
        const="all",
        nargs="?",
        choices=["fasta", "json", "all"],
    )
    parser.add_argument(
        "--predict_rotamers",
        action="store_true",
        default=True if rotamer_mode else False,
    )
    arg = parser.parse_args(args=arguments)
    output_paths = main_sample(arg)
    return pd.read_csv(output_paths[-1])


# }}}


# {{{ UI Sections
def _draw_output_section(
    selected_pdb,
    rotamer_mode,
    flat_dataset_map,
    flat_categories,
    pdb_to_probability,
    pdb_to_sequence,
    pdb_to_real_sequence,
):
    st.subheader(selected_pdb)
    try:
        pdb_session = show_pdb(selected_pdb[:4])
        showmol(pdb_session, height=500, width=640)
    except:
        pass
    # Show predicted sequence:
    st.subheader("Designed Sequence")
    st.code(pdb_to_sequence[selected_pdb])
    # Calculate Seq Metrics:
    real_metrics = _calculate_seq_metrics_wrapper(pdb_to_real_sequence[selected_pdb])
    predicted_metrics = _calculate_seq_metrics_wrapper(pdb_to_sequence[selected_pdb])
    similarity_score = _calculate_sequence_similarity_wrapper(
        pdb_to_real_sequence[selected_pdb], pdb_to_sequence[selected_pdb]
    )
    # Display original Metrics:
    st.write("Original Sequence Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Charge", f"{millify(real_metrics[0], precision=2)}")
    col2.metric("Isoelectric Point", f"{millify(real_metrics[1], precision=2)}")
    col3.metric("Molecular Weight", f"{millify(real_metrics[2], precision=2)}")
    col4.metric("Mol. Ext. Coeff. @ 280 nm", f"{millify(real_metrics[3], precision=2)}")
    # Display Predicted Metrics:
    st.write("Predicted Sequence Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Charge",
        f"{millify(predicted_metrics[0], precision=2)}",
        f"{millify(predicted_metrics[0] - real_metrics[0], precision=2)}",
    )
    col2.metric(
        "Isoelectric Point",
        f"{millify(predicted_metrics[1], precision=2)}",
        f"{millify(predicted_metrics[1] - real_metrics[1], precision=2)}",
    )
    col3.metric(
        "Molecular Weight",
        f"{millify(predicted_metrics[2], precision=2)}",
        f"{millify(predicted_metrics[2] - real_metrics[2], precision=2)}",
    )
    col4.metric(
        "Mol. Ext. Coeff. @ 280 nm",
        f"{millify(predicted_metrics[3], precision=2)}",
        f"{millify(predicted_metrics[3] - real_metrics[3], precision=2)}",
    )
    acc = accuracy_score(
        list(pdb_to_real_sequence[selected_pdb]), list(pdb_to_sequence[selected_pdb])
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Sequence Similarity",
        f"{millify(similarity_score * 100, precision=2)} %",
    )
    col3.metric("Sequence Identity", f"{millify(acc * 100, precision=2)} %")
    # Calculate composition of Sequence:
    comp_design = Counter(list(pdb_to_sequence[selected_pdb]))
    comp_real = Counter(list(pdb_to_real_sequence[selected_pdb]))
    new_comp = []
    for c_key, c_value in comp_real.items():
        current_value = ["Original", standard_amino_acids[c_key], c_value]
        new_comp.append(current_value)
    for c_key, c_value in comp_design.items():
        current_value = ["Designed", standard_amino_acids[c_key], c_value]
        new_comp.append(current_value)
    # Merge into Dataframe to allow for easy display with Altair:
    df = pd.DataFrame(new_comp, columns=["Source", "Residue", "# Qty"])
    chart_residue_comp = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            column=alt.Column(
                "Residue", title=None, header=alt.Header(orient="bottom")
            ),
            y=alt.Y(
                "# Qty",
                axis=alt.Axis(
                    labelAngle=0,
                ),
            ),
            x=alt.X("Source", axis=alt.Axis(ticks=True, labels=True, title="")),
            color=alt.Color("Source"),
            tooltip=["# Qty"],
        )
        .configure_view(stroke=None, strokeWidth=0.0)
    )
    # Show predicted probabilities:
    st.write("Predicted Probabilities")
    if len(pdb_to_sequence.keys()) > 1:
        slice_seq = {selected_pdb: pdb_to_sequence[selected_pdb]}
        slice_real = {selected_pdb: pdb_to_real_sequence[selected_pdb]}
    else:
        slice_seq = pdb_to_sequence
        slice_real = pdb_to_real_sequence
    _, real_seq_display = _encode_sequence_to_onehot(slice_seq, slice_real)
    real_seq_display = np.array(real_seq_display, dtype=str)
    real_seq_display = [w.replace("1.0", "Ori") for w in real_seq_display.ravel()]
    x, y = np.meshgrid(
        list(flat_categories), range(0, len(pdb_to_probability[selected_pdb]))
    )
    source_dict = {
        "Position": y.ravel(),
        "Residues": x.ravel(),
        "Probability (%)": np.array(pdb_to_probability[selected_pdb]).ravel() * 100,
    }
    if not rotamer_mode:
        source_dict["res"] = np.array(real_seq_display).ravel()
    source = pd.DataFrame(source_dict)
    # Rotamer Matrix is very large so it is hidden under a "spoiler" dropdown menu
    if rotamer_mode:
        altair_labels = _create_altair_labels(flat_categories)
        cm = (
            alt.Chart(source)
            .mark_rect()
            .encode(
                x=alt.X("Position:O"),
                y=alt.Y("Residues:O"),
                color="Probability (%):Q",
                tooltip=["Probability (%)", "Residues", "Position"],
            )
        )
        with st.expander("See Predicted Probabilities (Very Large Chart)"):
            st.altair_chart(cm, use_container_width=False)
    else:
        cm = (
            alt.Chart(source)
            .mark_rect()
            .encode(
                x=alt.X("Position:O"),
                y=alt.Y("Residues:O"),
                color="Probability (%):Q",
                tooltip=alt.Tooltip(["Probability (%)", "Residues", "Position"]),
            )
        )
        text = cm.mark_text(baseline="middle").encode(
            text="res:N",
            color=alt.condition(
                alt.datum.res == "Ori",
                alt.value("red"),
                alt.value(""),
            ),
        )
        cm_text = cm + text
        st.altair_chart(cm_text, use_container_width=False)
        st.write(
            '<p style="color:Tomato;">"ORI" indicates the residue in the original sequence.</p>',
            unsafe_allow_html=True,
        )
    if len(selected_pdb) == 5:
        current_chain = selected_pdb[-1]
        selected_dataset_map = flat_dataset_map[flat_dataset_map[:, 1] == current_chain]
    else:
        selected_dataset_map = flat_dataset_map
    # Build string datasetmap for selection
    f_i = np.core.defchararray.add(
        "(", np.array(np.arange(len(selected_dataset_map[:, 2])), dtype=str)
    )
    f_n = np.core.defchararray.add(f_i, ") ")
    f_0 = np.core.defchararray.add(f_n, selected_dataset_map[:, 2])
    f_1 = np.core.defchararray.add(f_0, " Chain ")
    f_2 = np.core.defchararray.add(f_1, selected_dataset_map[:, 1])
    f_3 = np.core.defchararray.add(f_2, " ")
    f_4 = np.core.defchararray.add(f_3, selected_dataset_map[:, 3])
    datamap_to_idx = dict(zip(f_4, range(len(f_4))))
    option = st.selectbox(
        "Explore probabilities at specific positions:", (f_4), key="option"
    )
    if "reload" in st.session_state.keys():
        pdb_session2 = show_pdb(selected_pdb[:4], st.session_state.option)
        showmol(pdb_session2, height=500, width=500)
        idx_pos = datamap_to_idx[st.session_state.option]
        vals = pdb_to_probability[selected_pdb][idx_pos]
        df = pd.DataFrame(vals)
        df.fillna(0, inplace=True)
        df.index = flat_categories
        st.subheader(f"Probability Distribution at position {st.session_state.option}")
        st.bar_chart(df, use_container_width=False)
    # Plot Residue Composition:
    st.write("Residue Composition")
    st.altair_chart(chart_residue_comp, use_container_width=False)
    return slice_seq, slice_real, real_metrics


def _draw_performance_section(selected_pdb, slice_seq, slice_real, res, axis_labels):
    """
    Draws performance section:
        - Descriptive Metrics
        - Prediction Bias
        - Confusion Matrix
    """
    # Plot Performance Metrics:
    st.title(f"Performance Metrics {selected_pdb}")
    results_dict = _calculate_metrics_wrapper(slice_seq, slice_real)
    st.subheader("Descriptive Metrics")
    cols = st.columns(4)
    # Display Accuracy:
    for i, c in enumerate(cols):
        acc_label = f"accuracy_{i + 2}"
        acc = results_dict[acc_label]
        c.metric(f"Top {i + 2} Accuracy", f"{millify(acc * 100, precision=2)} %")
    col1, col2, col3, _ = st.columns(4)
    col1.metric(
        f"Macro Precision",
        f"{millify(results_dict['precision'] * 100, precision=2)} %",
    )
    col2.metric(
        f"Macro Recall",
        f"{millify(results_dict['recall'] * 100, precision=2)} %",
    )
    # Plot Precision, Recall and F1:
    df = pd.DataFrame.from_dict(results_dict["report"])
    # Older version of scikit learn does not allow this:
    df.drop(
        ["accuracy", "micro avg", "macro avg", "weighted avg"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    df.drop(["support"], axis=0, inplace=True)
    df.columns = res
    st.bar_chart(df.T)
    # Plot Bias:
    st.subheader("Prediction Bias")
    vals = list(results_dict["bias"].values())
    df = pd.DataFrame(vals)
    df.index = res
    st.bar_chart(df)
    # Plot Confusion matrix:
    length_cm = len(results_dict["unweighted_cm"])
    x, y = np.meshgrid(range(0, length_cm), range(0, length_cm))
    z = results_dict["unweighted_cm"]
    # Convert this grid to columnar data expected by Altair
    source = pd.DataFrame(
        {
            "Predicted Residue": x.ravel(),
            "True Residue": y.ravel(),
            "Percentage (%)": z.ravel() * 100,
        }
    )
    cm = (
        alt.Chart(source)
        .mark_rect()
        .encode(
            x=alt.X("Predicted Residue:O", axis=alt.Axis(labelExpr=axis_labels)),
            y=alt.Y("True Residue:O", axis=alt.Axis(labelExpr=axis_labels)),
            color="Percentage (%):Q",
            tooltip=["Percentage (%)"],
        )
    )
    st.subheader("Confusion Matrix")
    st.altair_chart(cm, use_container_width=True)


def _draw_optimisation_section(
    selected_pdb,
    rotamer_mode,
    model,
    sample_n,
    workers,
    temperature,
    real_metrics,
    pdb_to_real_sequence,
):
    """
    Optimised Sequences using monte carlo.

    Samples `sample_n` sequences and looks for lowest difference in:
        - Charge
        - Isoelectric Point
        - Molecular Weight
        - Extinction Coefficient
        - Sequence Similarity
    """
    base = f"{model}{selected_pdb[:4]}"
    path_to_datasetmap = base + ".txt"
    if rotamer_mode:
        base += "_rot"
    path_to_pred_matrix = base + ".csv"
    opt_seq_metrics = _optimize_seq_with_montecarlo(
        path_to_pred_matrix,
        path_to_datasetmap,
        rotamer_mode,
        sample_n,
        workers,
        temperature,
    )
    sum_all_errors = False
    selected_columns = [
        "charge",
        "isoelectric_point",
        "molecular_weight",
        "molar_extinction",
    ]
    for c_idx, curr_col in enumerate(selected_columns):
        opt_seq_metrics[curr_col + "_mae"] = (
            opt_seq_metrics[curr_col] - real_metrics[c_idx]
        )
        opt_seq_metrics[curr_col + "_mae_norm"] = (
            opt_seq_metrics[curr_col] - opt_seq_metrics[curr_col].min()
        ) / (opt_seq_metrics[curr_col].max() - opt_seq_metrics[curr_col].min())
        if isinstance(sum_all_errors, np.ndarray):
            sum_all_errors += opt_seq_metrics[curr_col + "_mae_norm"].to_numpy()
        else:
            sum_all_errors = opt_seq_metrics[curr_col + "_mae_norm"].to_numpy()
    opt_seq_metrics["summed_mae"] = sum_all_errors
    opt_seq_metrics.sort_values("summed_mae", inplace=True)
    st.title(f"Optimized Sequence {selected_pdb}")
    for seq in range(0, 3):
        curr_slice = opt_seq_metrics.iloc[[seq]].values.tolist()[0]
        curr_sequence = curr_slice[1]
        st.subheader(f"Sequence {seq}")
        st.code(curr_sequence)
        curr_slice = curr_slice[2:]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Charge",
            f"{millify(curr_slice[0], precision=2)}",
            f"{millify(curr_slice[0] - real_metrics[0], precision=2)}",
        )
        col2.metric(
            "Isoelectric Point",
            f"{millify(curr_slice[1], precision=2)}",
            f"{millify(curr_slice[1] - real_metrics[1], precision=2)}",
        )
        col3.metric(
            "Molecular Weight",
            f"{millify(curr_slice[2], precision=2)}",
            f"{millify(curr_slice[2] - real_metrics[2], precision=2)}",
        )
        col4.metric(
            "Mol. Ext. Coeff. @ 280 nm",
            f"{millify(curr_slice[3], precision=2)}",
            f"{millify(curr_slice[3] - real_metrics[3], precision=2)}",
        )
        acc = accuracy_score(
            list(curr_sequence), list(pdb_to_real_sequence[selected_pdb])
        )
        similarity_score = _calculate_sequence_similarity_wrapper(
            curr_sequence, pdb_to_real_sequence[selected_pdb]
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Sequence Similarity",
            f"{millify(similarity_score * 100, precision=2)} %",
        )
        col3.metric("Sequence Identity", f"{millify(acc * 100, precision=2)} %")
    st.subheader("Sampled Sequences")
    st.write(opt_seq_metrics)
    st.download_button(
        label="Download data as CSV",
        data=opt_seq_metrics.to_csv().encode("utf-8"),
        file_name=f"monte_carlo_{model}_{selected_pdb}.csv",
        mime="text/csv",
    )


def _draw_sidebar(all_pdbs: t.List[str]):
    """
    Draws Sidebar with elements:
        - PDB Code Input
        - Model Dropdown
        - Advanced Settings
            - Monte Carlo
            - Number of Sequences to Sample
            - Temperature Factor
    """
    st.sidebar.title("Design Proteins")
    pdb = st.sidebar.text_input("Enter a PDB Code:", value="1qys", placeholder="1qys")
    pdb = pdb.lower()
    st.sidebar.write("or")
    dataset1 = st.sidebar.file_uploader(
        label="Upload your backbone/PDB of interest", disabled=True
    )
    model = st.sidebar.selectbox(
        label="Choose your Model",
        options=(
            "TIMED",
            "TIMED_Deep",
            "TIMED_rotamer",
            "TIMED_rotamer_balanced",
            "TIMED_rotamer_not_so_deep",
            "TIMED_rotamer_deep",
            "DenseCPD",
            "DenseNet",
            "ProDCoNN",
        ),
        help="To check the performance of each of the models visit: https://github.com/wells-wood-research/timed-design/releases/tag/model",
    )
    # Add Advanced settings menu for monte carlo sampling
    with st.sidebar.expander("Advanced Settings"):
        # Not using the sidebar as per https://github.com/streamlit/streamlit/issues/3157
        use_montecarlo_button = st.empty()
        use_montecarlo = use_montecarlo_button.checkbox(
            "Optimize sequences using Monte Carlo", key="mc"
        )
        sample_n_button = st.empty()
        sample_n = sample_n_button.slider("Number of sequences to sample", 3, 300, 200)
        temperature_button = st.empty()
        temperature = temperature_button.slider("Temperature Factor", 0.1, 1.0, 1.0)
    placeholder_run_button = st.sidebar.empty()
    result = placeholder_run_button.button("Run model", key="1")
    st.sidebar.markdown(
        "[Tell us what you think!](https://forms.office.com/Pages/ResponsePage.aspx?id=sAafLmkWiUWHiRCgaTTcYY_RqhHaishKsB4CsyQgPCxUOU9DQjhJU0s1QjZVVTNPU0xDVzlFTEhNMS4u)"
    )
    if pdb not in all_pdbs:
        st.sidebar.error("PDB code not found")
        placeholder_run_button.button("Run model", disabled=True, key="4")
    return (
        model,
        result,
        pdb,
        (
            placeholder_run_button,
            use_montecarlo_button,
            sample_n_button,
            temperature_button,
        ),
        (use_montecarlo, sample_n, temperature),
    )
# }}}


def main(args):
    path_to_data = Path(args.path_to_data)
    path_to_models = Path(args.path_to_models)
    path_to_pdb = Path(args.path_to_pdb)
    # Check path exists:
    assert (
        path_to_data.exists()
    ), f"Path to data {path_to_data} does not exists. Set it up in the argparse."
    assert (
        path_to_models.exists()
    ), f"Path to models {path_to_models} does not exists. Set it up in the argparse."
    assert (
        path_to_pdb.exists()
    ), f"Path to pdb {path_to_pdb} does not exists.Set it up in the argparse."
    # Find all pdbs:
    all_pdbs = _search_all_pdbs(path_to_pdb)
    # Draw Sidebar:
    (
        model,
        result,
        pdb,
        (
            placeholder_run_button,
            use_montecarlo_button,
            sample_n_button,
            temperature_button,
        ),
        (use_montecarlo, sample_n, temperature),
    ) = _draw_sidebar(all_pdbs)
    # Find selected model
    model_path = path_to_models / (model + ".h5")
    res = list(standard_amino_acids.values())
    axis_labels = _create_altair_labels(res)
    # Start app:
    if result or "reload" in st.session_state.keys():
        # When user clicks on calculate, check that the model is a rotamer model or not:
        rotamer_mode = True if "rotamer" in model else False
        if rotamer_mode:
            _, flat_categories = _get_rotamer_codec_wrapper()
        else:
            flat_categories = standard_amino_acids.values()
        # Disable Run Model button while running the app: (avoids clogging)
        placeholder_run_button.button("Run model", disabled=True, key="2")
        with st.sidebar.expander("Advanced Settings"):
            use_montecarlo_button.checkbox(
                "Optimize sequences using Monte Carlo", disabled=True, key="mc_2"
            )
            sample_n_button.slider(
                "Number of sequences to sample", 3, 300, 200, disabled=True, key="sn_2"
            )
            temperature_button.slider(
                "Temperature Factor", 0.1, 1.0, 1.0, disabled=True, key="t_2"
            )
        with st.spinner("Voxelising Protein Structure..."):
            t0_apo = time.time()
            dataset = _build_aposteriori_dataset_wrapper(
                path_to_pdb=path_to_pdb,
                pdb_code=pdb,
                output_path=path_to_data,
                workers=args.workers,
            )
            t1_apo = time.time()
        # Use model to predict:
        t0 = time.time()
        (
            flat_dataset_map,
            pdb_to_sequence,
            pdb_to_probability,
            pdb_to_real_sequence,
            _,
            _,
        ) = predict_dataset(dataset, model_path, rotamer_mode, pdb)
        t1 = time.time()
        time_string = time.strftime("%M m %S s", time.gmtime(t1 - t0))
        apo_time_string = time.strftime("%M m %S s", time.gmtime(t1_apo - t0_apo))
        total_time_string = time.strftime("%M m %S s", time.gmtime(t1 - t0_apo))
        if "count" not in st.session_state.keys():
            st.success(
                f"Done! Took {total_time_string} in total. Voxelisation took {apo_time_string} and prediction took {time_string}"
            )
        # Print Results:
        st.title("Model Output")
        # For each key in the dataset:
        for k in pdb_to_probability.keys():
            slice_seq, slice_real, real_metrics = _draw_output_section(
                k,
                rotamer_mode,
                flat_dataset_map,
                flat_categories,
                pdb_to_probability,
                pdb_to_sequence,
                pdb_to_real_sequence,
            )
            _draw_performance_section(k, slice_seq, slice_real, res, axis_labels)
            if "mc_3" in st.session_state.keys():
                if st.session_state["mc_3"]:
                    _draw_optimisation_section(
                        k,
                        rotamer_mode,
                        model,
                        sample_n,
                        args.workers,
                        temperature,
                        real_metrics,
                        pdb_to_real_sequence,
                    )

        placeholder_run_button.button("Run model", disabled=False, key="3")
        with st.sidebar.expander("Advanced Settings"):
            use_montecarlo_button.checkbox(
                "Optimize sequences using Monte Carlo", disabled=False, key="mc_3"
            )
            sample_n_button.slider(
                "Number of sequences to sample", 3, 300, 200, disabled=False, key="sn_3"
            )
            temperature_button.slider(
                "Temperature Factor", 0.1, 1.0, 1.0, disabled=False, key="t_3"
            )
        # Only show specific plots after interaction with the interface
        if "reload" not in st.session_state:
            st.session_state.reload = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path_to_models", type=str, help="Path to .h5 model files")
    parser.add_argument("--path_to_pdb", type=str, help="Path to pdb folder")
    parser.add_argument("--path_to_data", type=str, help="Path to .hdf5 data folder")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers to use (default: 8)"
    )
    params = parser.parse_args()
    main(params)
