#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j32
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/ATD/out/cifar10-svhn-Balanced/32.log

python test_ATD.py  --run_name cifar10-svhn-Balanced-32 --model_type "fea" --in_dataset cifar10-svhn --batch_size 256 --eps 0.0313 --attack_iters 100 \
            --out_datasets cifar10-svhn
