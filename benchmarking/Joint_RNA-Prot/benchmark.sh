#!/bin/bash

#SBATCH -J RF  
#SBATCH --output=%x_%j.out 
#SBATCH --error=%x_%j.err     
#SBATCH --exclude=gpu[001-004]
#SBATCH --mem=80G
#SBATCH --time=10:00:00

python RF.py \
  --training_data_rna_fasta="" \
  --training_data_protein_fasta="" \
  --Model2_train="" \
  --test_data_rna_fasta="" \
  --test_data_protein_fasta="" \
  --Model2_test="" \
  --data_set="" \
  --batch_size=
