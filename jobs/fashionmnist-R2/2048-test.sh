#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j2048
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/ATD/out/fashionmnist-R2-Imbalanced/2048.log

python test_ATD.py  --run_name fashionmnist-R2-Imbalanced-2048 --model_type "fea" --in_dataset fashionmnist-R2 --batch_size 256 --eps 0.0313 --attack_iters 100 \
            --out_datasets fashionmnist-R2
