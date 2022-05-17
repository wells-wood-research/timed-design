import tempfile
from pathlib import Path

import streamlit as st

from utils.utils import load_dataset_and_predict


def predict_dataset(file, path_to_model):
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
            predict_rotamers=True,
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
    st.sidebar.title("Show Proteins")
    dataset = st.sidebar.file_uploader(label="Choose your Dataset")
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
    if result:
        placeholder.button("Run model", disabled=True, key="2")
        with st.spinner("Calculating results.."):
            (
                flat_dataset_map,
                pdb_to_sequence,
                pdb_to_probability,
                pdb_to_real_sequence,
                pdb_to_consensus,
                pdb_to_consensus_prob,
            ) = predict_dataset(dataset, model_path)
        st.success("Done!")
        st.title("Dataset Map")
        st.write(flat_dataset_map)
        st.title("Predicted Sequences")
        st.write(pdb_to_sequence)
        st.title("Predicted Probabilities")
        st.write(pdb_to_probability)
        st.title("Real Sequences")
        st.write(pdb_to_real_sequence)
        placeholder.button("Run model", disabled=False, key="3")


if __name__ == "__main__":
    main()

# style = st.sidebar.selectbox('style',['line','cross','stick','sphere','cartoon','clicksphere'])
# xyzview = py3Dmol.view(query='pdb:'+protein)
# xyzview.setStyle({style:{'color':'spectrum'}})
# xyzview.setBackgroundColor(bcolor)
# showmol(xyzview, height = 500,width=800)
