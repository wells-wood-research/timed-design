import tempfile
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from ampal.amino_acids import standard_amino_acids
from millify import millify
from sklearn.metrics import accuracy_score

from utils.analyse_utils import calculate_metrics, calculate_seq_metrics
from utils.utils import get_rotamer_codec, load_dataset_and_predict


def predict_dataset(file, path_to_model, rotamer_mode):
    with tempfile.NamedTemporaryFile(delete=True) as dataset_file:
        dataset_file.write(file.getbuffer())
        dataset_file.seek(0)  # Resets the buffer back to the first line
        path_to_dataset = Path(dataset_file.name)
        (
            flat_dataset_map,
            pdb_to_sequence,
            pdb_to_probability,
            pdb_to_real_sequence,
            pdb_to_consensus,
            pdb_to_consensus_prob,
        ) = load_dataset_and_predict(
            [path_to_model],
            path_to_dataset,
            batch_size=12,
            start_batch=0,
            blacklist=None,
            dataset_map_path="data.txt",
            predict_rotamers=rotamer_mode,
        )
    return (
        flat_dataset_map,
        pdb_to_sequence,
        pdb_to_probability,
        pdb_to_real_sequence,
        pdb_to_consensus,
        pdb_to_consensus_prob,
    )


def main():
    path_to_models = Path("models")
    st.sidebar.title("Design Proteins")
    dataset = st.sidebar.file_uploader(label="Choose your PDB of interest")
    model = st.sidebar.selectbox(
        label="Choose your Model",
        options=(
            "TIMED",
            "TIMED_Deep",
            "TIMED_not_so_deep",
            "TIMED_rotamer",
            "TIMED_rotamer_deep",
            "DenseCPD",
            "DenseNet",
            "ProDCoNN",
        ),
        help="To check the performance of each of the models visit: https://github.com/wells-wood-research/timed-design/releases/tag/model",
    )
    model_path = path_to_models / (model + ".h5")
    placeholder = st.sidebar.empty()
    result = placeholder.button("Run model", key="1")
    res = list(standard_amino_acids.values())
    axis_labels = f"""
            datum.label == 0 ? '{res[0]}'
            :datum.label == 1 ? '{res[1]}'
            :datum.label == 2 ? '{res[2]}'
            :datum.label == 3 ? '{res[3]}'
            :datum.label == 4 ? '{res[4]}'
            :datum.label == 5 ? '{res[5]}'
            :datum.label == 6 ? '{res[6]}'
            :datum.label == 7 ? '{res[7]}'
            :datum.label == 8 ? '{res[8]}'
            :datum.label == 9 ? '{res[9]}'
            :datum.label == 10 ? '{res[10]}'
            :datum.label == 11 ? '{res[11]}'
            :datum.label == 12 ? '{res[12]}'
            :datum.label == 13 ? '{res[13]}'
            :datum.label == 14 ? '{res[14]}'
            :datum.label == 15 ? '{res[15]}'
            :datum.label == 16 ? '{res[16]}'
            :datum.label == 17 ? '{res[17]}'
            :datum.label == 18 ? '{res[18]}'
            : '{res[19]}'
            """
    if result:
        rotamer_mode = True if "rotamer" in model else False
        if rotamer_mode:
            _, flat_categories = get_rotamer_codec()
        else:
            flat_categories = standard_amino_acids.values()
        placeholder.button("Run model", disabled=True, key="2")
        with st.spinner("Calculating results.."):
            (
                flat_dataset_map,
                pdb_to_sequence,
                pdb_to_probability,
                pdb_to_real_sequence,
                _,
                _,
            ) = predict_dataset(dataset, model_path, rotamer_mode)
        st.success("Done!")
        st.title("Model Output")
        for k in pdb_to_probability.keys():
            st.subheader(k)
            st.write("Designed Sequence")
            st.code(pdb_to_sequence[k])
            comp_design = Counter(list(pdb_to_sequence[k]))
            comp_real = Counter(list(pdb_to_real_sequence[k]))
            real_metrics = calculate_seq_metrics(pdb_to_real_sequence[k])
            predicted_metrics = calculate_seq_metrics(pdb_to_sequence[k])
            st.write("Original Sequence Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Charge", f"{millify(real_metrics[0], precision=2)}")
            col2.metric("Isoelectric Point", f"{millify(real_metrics[1], precision=2)}")
            col3.metric("Molecular Weight", f"{millify(real_metrics[2], precision=2)}")
            st.write("Predicted Sequence Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Charge",
                f"{millify(predicted_metrics[0], precision=2)}",
                f"{millify(predicted_metrics[0]-real_metrics[0], precision=2)}",
            )
            col2.metric(
                "Isoelectric Point",
                f"{millify(predicted_metrics[1], precision=2)}",
                f"{millify(predicted_metrics[1]-real_metrics[1], precision=2)}",
            )
            col3.metric(
                "Molecular Weight",
                f"{millify(predicted_metrics[2], precision=2)}",
                f"{millify(predicted_metrics[2]-real_metrics[2], precision=2)}",
            )
            acc = accuracy_score(
                list(pdb_to_real_sequence[k]), list(pdb_to_sequence[k])
            )
            st.metric("Sequence Identity", f"{millify(acc*100, precision=2)} %")
            new_comp = []
            for c_key, c_value in comp_real.items():
                current_value = ["Original", standard_amino_acids[c_key], c_value]
                new_comp.append(current_value)
            for c_key, c_value in comp_design.items():
                current_value = ["Designed", standard_amino_acids[c_key], c_value]
                new_comp.append(current_value)
            df = pd.DataFrame(new_comp, columns=["Source", "Residue", "# Qty"])
            chart = (
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
            st.write("Residue Composition")
            st.altair_chart(chart, use_container_width=False)
            st.write("Predicted Probabilities")
            x, y = np.meshgrid(
                list(flat_categories), range(0, len(pdb_to_probability[k]))
            )
            source = pd.DataFrame(
                {
                    "Position": y.ravel(),
                    "Residues": x.ravel(),
                    "Probability (%)": np.array(pdb_to_probability[k]).ravel() * 100,
                }
            )

            if rotamer_mode:
                rot_labels = ""
                for i, cat in enumerate(flat_categories):
                    if i == 0:
                        base = "datum.label == "
                    else:
                        base = ":datum.label == "
                    if i == len(flat_categories) - 1:
                        full = f": '{cat}'"
                    else:
                        full = base + str(i) + f" ? '{cat}'\n"
                    rot_labels += full
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
            else:
                cm = (
                    alt.Chart(source)
                    .mark_rect()
                    .encode(
                        x=alt.X("Position:O"),
                        y=alt.Y("Residues:O"),
                        color="Probability (%):Q",
                        tooltip=alt.Tooltip(
                            ["Probability (%)", "Residues", "Position"]
                        ),
                    )
                )
            if rotamer_mode:
                with st.expander("See Predicted Probabilities (Very Large Chart)"):
                    st.altair_chart(cm, use_container_width=False)
            else:
                st.altair_chart(cm, use_container_width=False)
        st.title("Overall Performance Metrics")
        results_dict = calculate_metrics(pdb_to_sequence, pdb_to_real_sequence)
        st.subheader("Descriptive Metrics")
        cols = st.columns(4)
        for i, c in enumerate(cols):
            acc_label = f"accuracy_{i+2}"
            acc = results_dict[acc_label]
            c.metric(f"Top {i+2} Accuracy", f"{millify(acc*100, precision=2)} %")
        col1, col2, col3, _ = st.columns(4)
        col1.metric(
            f"Macro Precision",
            f"{millify(results_dict['precision'] * 100, precision=2)} %",
        )
        col2.metric(
            f"Macro Recall", f"{millify(results_dict['recall'] * 100, precision=2)} %"
        )
        col3.metric(
            f"AUC OVO", f"{millify(results_dict['auc_ovo'] * 100, precision=2)} %"
        )
        df = pd.DataFrame.from_dict(results_dict["report"])
        df.drop(["accuracy", "macro avg", "weighted avg"], axis=1, inplace=True)
        df.drop(["support"], axis=0, inplace=True)
        df.columns = res
        st.bar_chart(df.T)
        st.subheader("Prediction Bias")
        vals = list(results_dict["bias"].values())
        df = pd.DataFrame(vals)
        df.fillna(0, inplace=True)
        df.index = res
        st.bar_chart(df)
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
        placeholder.button("Run model", disabled=False, key="3")


if __name__ == "__main__":
    main()

# style = st.sidebar.selectbox('style',['line','cross','stick','sphere','cartoon','clicksphere'])
# xyzview = py3Dmol.view(query='pdb:'+protein)
# xyzview.setStyle({style:{'color':'spectrum'}})
# xyzview.setBackgroundColor(bcolor)
# showmol(xyzview, height = 500,width=800)
