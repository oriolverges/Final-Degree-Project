#!/bin/bash

#SBATCH -J Model2_Embed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=00:40:00


GPU_INDEXES="$SLURM_JOB_GPUS"
NUM_GPUS=$(echo "$GPU_INDEXES" | tr ',' '\n' | grep -c '[0-9]')

python embeddings.py \
    --data_path ""\
    --weights_path "" \
    --embedding_dir "" \
    --output_name "" \
    --emsize 256 --nhead 8 \
    --nhid 512 --nlayers 4 --dropout 0.2 \
    --batch_size 32 \
    --devices $NUM_GPUS
