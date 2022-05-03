# Scripts

These are a collection of scripts used to automate analysis. You may need to change paths.

Below a description of each of them:

## Running Rotamer Analysis 

**File**: `run_rotamer_analysis.sh`
**Description**: 

Runs a rotamer analysis on a set of PDBs. Then performs four sets of analyses.

Analysis 1: TIMED_rotamer vs real rotamers from crystal structure
- ROC AUC score (OVO)
- Classification report (https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
- Accuracy (top 1, 2, 3, 4, 5 accuracy)
- Macro Precision and Recall
- Prediction bias (https://developers.google.com/machine-learning/crash-course/classification/prediction-bias)
- Weighted / Unweighted confusion matrix

Analysis 2: TIMED_rotamer vs TIMED_rotamer sequence put through SCWRL
Calculates the metrics above but comparing the model prediction put through SCWRL

Analysis 3: TIMED_rotamer vs Real sequence from crystal put through SCWRL
Calculates the metrics above but comparing the model prediction vs the real sequence put through SCWRL

Then saves metrics to a file. It also saves SCWRL_scores for the predicted and the real sequences.


## Running Monte Carlo Sampling 

**File**: `run_sampling.sh`
**Description**: 

Runs a Monte Carlo sampling on 59 structures used in the PDBench Paper. We apply a temperature factor [0.1, 0.5, 1, 2, 5] and sample 200 new sequences for each structure.

We do this for all the models deep and default in 20 (residues) or 338 (rotamers) classes.


## Running Netsolp

**File**: `run_netsolp.sh`
**Description**: 

Runs the NetSolP-1.0 model for the .fasta structures sampled by Monte Carlo. We use the Distilled Model and predict both Solubility (S) and Expressivity (U).




