python analyse_rotamers.py --path_to_pred_matrix data/rotamer/TIMED_not_so_deep_rot.csv --path_to_datasetmap data/rotamer/TIMED_not_so_deep_rot.txt --workers 10 --path_to_pdb pdb/ --output_path rotamer_analysis
python analyse_rotamers.py --path_to_pred_matrix data/rotamer/TIMED_rot.csv --path_to_datasetmap data/rotamer/TIMED_rot.txt --workers 10 --path_to_pdb pdb/ --output_path rotamer_analysis_TIMED_rot
python analyse_rotamers.py --path_to_pred_matrix data/rotamer/TIMED_rotamer_deep_rot.csv --path_to_datasetmap data/rotamer/TIMED_rotamer_deep_rot.txt --workers 10 --path_to_pdb pdb/ --output_path rotamer_analysis