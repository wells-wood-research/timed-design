#!/bin/bash
#SBATCH -A CSD3
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 36:00:00
#SBATCH --mail-type=ALL

# load appropriate modules
source ~/rds/hpc-work/conda/bin/conda
source ~/.bashrc
source ~/.bash_profile
module load rhel8/default-amp
module load alphafold

# point to location of AlphaFold data
DATA=/rds/project/rds-dMMtPvqHWv4/data/

run_alphafold \
--pdb70_database_path=/data/pdb70/pdb70 \
--bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--uniclust30_database_path /data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
--output_dir  $PWD \
--fasta_paths $PWD/TIMED_1.fasta,$PWD/TIMED_2.fasta,$PWD/TIMED_3.fasta,$PWD/TIMED_4.fasta,$PWD/TIMED_5.fasta,$PWD/TIMED_6.fasta,$PWD/TIMED_7.fasta,$PWD/TIMED_8.fasta,$PWD/TIMED_9.fasta,$PWD/TIMED_10.fasta,$PWD/TIMED_11.fasta,$PWD/TIMED_12.fasta,$PWD/TIMED_13.fasta,$PWD/TIMED_14.fasta,$PWD/TIMED_15.fasta,$PWD/TIMED_16.fasta,$PWD/TIMED_17.fasta,$PWD/TIMED_18.fasta,$PWD/TIMED_19.fasta,$PWD/TIMED_20.fasta,$PWD/TIMED_21.fasta,$PWD/TIMED_22.fasta,$PWD/TIMED_23.fasta,$PWD/TIMED_24.fasta,$PWD/TIMED_25.fasta,$PWD/TIMED_26.fasta,$PWD/TIMED_27.fasta,$PWD/TIMED_28.fasta,$PWD/TIMED_29.fasta,$PWD/TIMED_30.fasta,$PWD/TIMED_31.fasta,$PWD/TIMED_32.fasta,$PWD/TIMED_33.fasta,$PWD/TIMED_34.fasta,$PWD/TIMED_35.fasta,$PWD/TIMED_36.fasta,$PWD/TIMED_37.fasta,$PWD/TIMED_38.fasta,$PWD/TIMED_39.fasta,$PWD/TIMED_40.fasta,$PWD/TIMED_41.fasta,$PWD/TIMED_42.fasta,$PWD/TIMED_43.fasta,$PWD/TIMED_44.fasta,$PWD/TIMED_45.fasta,$PWD/TIMED_46.fasta,$PWD/TIMED_47.fasta,$PWD/TIMED_48.fasta,$PWD/TIMED_49.fasta,$PWD/TIMED_50.fasta,$PWD/TIMED_51.fasta,$PWD/TIMED_52.fasta,$PWD/TIMED_53.fasta,$PWD/TIMED_54.fasta,$PWD/TIMED_55.fasta,$PWD/TIMED_56.fasta,$PWD/TIMED_57.fasta,$PWD/TIMED_58.fasta,$PWD/TIMED_59.fasta,$PWD/TIMED_60.fasta,$PWD/TIMED_61.fasta,$PWD/TIMED_62.fasta,$PWD/TIMED_63.fasta,$PWD/TIMED_64.fasta,$PWD/TIMED_65.fasta,$PWD/TIMED_66.fasta,$PWD/TIMED_67.fasta,$PWD/TIMED_68.fasta,$PWD/TIMED_69.fasta,$PWD/TIMED_70.fasta,$PWD/TIMED_71.fasta,$PWD/TIMED_72.fasta,$PWD/TIMED_73.fasta,$PWD/TIMED_74.fasta,$PWD/TIMED_75.fasta,$PWD/TIMED_76.fasta,$PWD/TIMED_77.fasta,$PWD/TIMED_78.fasta,$PWD/TIMED_79.fasta,$PWD/TIMED_80.fasta \
--max_template_date=2020-05-14 \
--db_preset=full_dbs \
--use_gpu_relax=True \
--cpus 32