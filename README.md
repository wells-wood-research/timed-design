# timed-predict

## Setting up the environment:

1. Setting up conda:

```
conda create --name timed_predict python=3.8
```

```
conda activate timed_predict
```

### Easy Install:

```
sh setup.sh
```

### Manual Install:


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

## Using the models for predicting

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

### Predicting Rotamers:

In order to use a rotamer model, use the flag `--predict_rotamers True`:


```
python3 predict.py --path_to_dataset dataset.hdf5 --path_to_model timed_rot.h5 --predict_rotamers True
```


