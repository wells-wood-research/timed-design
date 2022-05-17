import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from ampal.amino_acids import standard_amino_acids
from millify import millify
from sklearn.metrics import accuracy_score

from utils.analyse_utils import calculate_metrics_, calculate_seq_metrics
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
    rotamer_mode = True if "rotamer" in model else False
    if rotamer_mode:
        _, flat_categories = get_rotamer_codec()
    else:
        flat_categories = standard_amino_acids.values()
    placeholder = st.sidebar.empty()
    result = placeholder.button("Run model", key="1")
    if result:
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
            st.write("Predicted Probabilities")

            df = pd.DataFrame(pdb_to_probability[k], columns=flat_categories)
            st.dataframe(df.style.highlight_max(axis=1))

        st.title("Performance")
        results_dict = calculate_metrics_(
            pdb_to_sequence, pdb_to_real_sequence
        )
        st.subheader("Overall Performance")
        st.write(results_dict["auc_ovo"])
        st.write(results_dict["report"])
        st.write(results_dict["accuracy_1"])
        st.write(results_dict["accuracy_2"])
        st.write(results_dict["accuracy_3"])
        st.write(results_dict["accuracy_4"])
        st.write(results_dict["accuracy_5"])
        st.write(results_dict["precision"])
        st.write(results_dict["recall"])
        st.write(results_dict["count_labels"])
        st.write(results_dict["count_pred"])
        st.write(results_dict["bias"])
        st.write(results_dict["unweighted_cm"])

        for k in pdb_to_probability.keys():
            st.subheader(k)
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
        placeholder.button("Run model", disabled=False, key="3")


if __name__ == "__main__":
    main()

# style = st.sidebar.selectbox('style',['line','cross','stick','sphere','cartoon','clicksphere'])
# xyzview = py3Dmol.view(query='pdb:'+protein)
# xyzview.setStyle({style:{'color':'spectrum'}})
# xyzview.setBackgroundColor(bcolor)
# showmol(xyzview, height = 500,width=800)
