# timed-predict

## Setting up the environment:

1. Setting up conda:

```
conda create --name timed_predict python=3.7
```

```
conda activate timed_predict
```

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

## Using the models for predicting

1. Make a folder with all the pdb files you want to predict

*Note:* Please use the same format for all the structures

2. Create the dataset using aposteriori

```shell
make-frame-dataset YOUR_PDB_FOLDER  -e YOUR_PDB_EXTENSION --voxels-per-side 21 --frame-edge-length 21 -g True -p 6 -n dataset -v -r -cb True -ae CNOCBCA  --compression_gzip True -o .  --voxelise_all_states True
```

For more info about other options, please see https://github.com/wells-wood-research/aposteriori/

3. Download your model of interest from:

https://github.com/wells-wood-research/timed/releases

3. Finally run: 







