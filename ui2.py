import tempfile
from pathlib import Path

import streamlit as st

from utils.utils import load_dataset_and_predict


def predict_dataset(file, model):
    with tempfile.NamedTemporaryFile(
            delete=True
    ) as dataset_file:
        dataset_file.write(file.getbuffer())
        dataset_file.seek(0)  # Resets the buffer back to the first line
        path_to_dataset = Path(dataset_file.name)
        path_to_model = Path(model.name)
        st.write(path_to_dataset)
        st.write(path_to_dataset.exists())
        st.write(path_to_model)
        st.write(path_to_model.exists())
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
    st.title("Dataset Map")
    st.write(flat_dataset_map)
    st.title("Predicted Sequences")
    st.write(pdb_to_sequence)
    st.title("Predicted Probabilities")
    st.write(pdb_to_probability)
    st.title("Real Sequences")
    st.write(pdb_to_real_sequence)


def main():
    st.sidebar.title('Show Proteins')
    prot_str='1A2C,1BML,1D5M,1D5X,1D5Z,1D6E,1DEE,1E9F,1FC2,1FCC,1G4U,1GZS,1HE1,1HEZ,1HQR,1HXY,1IBX,1JBU,1JWM,1JWS'
    prot_list=prot_str.split(',')
    dataset = st.sidebar.file_uploader(label="Choose your Dataset")
    model = st.sidebar.file_uploader(label="Choose your Model")

    result = st.button('Run TIMED')
    if result:
        st.write('Calculating results...')
        predict_dataset(dataset, model)

if __name__ == '__main__':
    main()

# style = st.sidebar.selectbox('style',['line','cross','stick','sphere','cartoon','clicksphere'])
# xyzview = py3Dmol.view(query='pdb:'+protein)
# xyzview.setStyle({style:{'color':'spectrum'}})
# xyzview.setBackgroundColor(bcolor)
# showmol(xyzview, height = 500,width=800)