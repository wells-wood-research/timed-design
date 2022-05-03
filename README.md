<div align="center">
  <img src="logo.png"><br>
  <h2>Protein Sequence Design Made Easy</h2><br>
</div>

[timed-predict](https://github.com/wells-wood-research/timed-predict) is a library to use protein sequence design models and analyse predictions. We feature retrained Keras models for novel models (**TIMED** and **TIMED-rotamer**) as well as re-implementations of well known models for which code or model are not intuitively available (**ProDCoNN**, **DenseCPD**, **DenseNet**). 

Table of Contents:
- [1. Use Models](#1-use-models)
- [2. Sample Sequences Using Monte Carlo](#2-sample-sequences-using-monte-carlo)
- [3. Analyse Rotamer Predictions](#3-analyse-rotamer-predictions)
- [4. Cite This Work](#4-cite-this-work)

## 1. Use Models:

**File**: `predict.py`

**Description**:  

Use any model to predict a 3D structure. This requires a backbone in a .pdb structure. The side-chains of the residues will be automatically removed by [aposteriori](https://github.com/wells-wood-research/aposteriori), thus the prediction will be performed uniquely on the empty backbone. Your chosen model will attempt to predict which residues best fit the position and will return a `.fasta` file as well as a probability distribution in `.csv` format. 

### 1.1 Set up the environment:

1. Setting up conda:

```
conda create --name timed_predict python=3.8
```

```
conda activate timed_predict
```

#### Easy Install:

```
sh setup.sh
```

#### Manual Install:


2. Install poetry:

```
conda install poetry 
```

3. Install aposteriori (voxelisation of proteins)

```
git clone https://github.com/wells-wood-research/aposteriori.git
```

```
cd aposteriori
```

```
poetry install
```

You may have issues install cython, for which you should try installing it with conda:

```
conda install cython
```

Now install aposteriori with pip (when aposteriori will be published we may be able to use pypi)

```
pip install .
```

For GPU Support run:

```
conda install cudatoolkit cudnn cupti 
```

Move out of the `aposteriori` folder with `cd ..`. Then clone TIMED:

```shell
git clone https://github.com/wells-wood-research/timed.git
```


```shell
cd timed
```


```shell
poetry install
```


```shell
pip install tqdm
```

### 1.2 Using the models for predicting

1. Make a folder with all the pdb files you want to predict

*Note:* Please use the same format for all the structures

2. Create the dataset using aposteriori

```shell
make-frame-dataset YOUR_PDB_FOLDER  -e YOUR_PDB_EXTENSION --voxels-per-side 21 --frame-edge-length 21 -g True -p 6 -n dataset -v -r -cb True -ae CNOCBCA  --compression_gzip True -o .  --voxelise_all_states True
```

For more info about other options, please see https://github.com/wells-wood-research/aposteriori/

for a sample dataset use:

```shell
poetry run make-frame-dataset aposteriori/tests/testing_files/pdb_files/ -e .pdb --name data --voxels-per-side 21 --frame-edge-length 21 -p 8  -vrz -cb False -ae CNOCBCA -g True 
```


3. Download your model of interest from:

https://github.com/wells-wood-research/timed/releases

3. Finally run: 


```
python3 predict.py --path_to_dataset {DATASET_PATH}.hdf5 --path_to_model {MODEL_PATH}.h5
```

eg.

```
python3 predict.py --path_to_dataset dataset.hdf5 --path_to_model timed_2.h5
```

### 1.3 Predicting Rotamers:

In order to use a rotamer model, use the flag `--predict_rotamers True`:


```
python3 predict.py --path_to_dataset dataset.hdf5 --path_to_model timed_rot.h5 --predict_rotamers True
```




## 2. Sample Sequences Using Monte Carlo:

**File**: `sample.py`
**Description**:  

Uses Monte Carlo sampling to sample sequences from a  probability distribution. A temperature factor can be applied to affect the distributions. It will return a `.fasta` file and/or a `.json` file with the sequences and a `.csv` file with basic sequence metrics such as isoelectric point, molecular weight and charge. Further metrics can be calculated using NetSolP-1.0 (see `scripts/run_netsolp.sh`).

## 3. Analyse Rotamer Predictions:

**File**: `analyse_rotamers.py`
**Description**:  

---Under construction---

## 4. Cite This Work:

---Under construction---
