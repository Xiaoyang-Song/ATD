#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=ATD-SV
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/ATD/out/svhn.log


# python train_ATD.py  --run_name cifar10-svhn --model_type "fea" --training_type "adv" \
#     --in_dataset cifar10-svhn --alpha 0.5 --batch_size 128 --num_epochs 20 --eps 0.0313 --attack_iters 10 \
#     --regime Balanced --ood_dset cifar10-svhn --n_ood 64 --valsize 10000

python train_ATD.py  --run_name svhn --model_type "fea" --training_type "adv" \
    --in_dataset svhn --alpha 0.5 --batch_size 256 --num_epochs 20 --eps 0.0313 --attack_iters 10 \
    --regime Balanced --ood_dset svhn --n_ood 1024 --valsize 10000

python test_ATD.py  --run_name svhn --model_type "fea" --in_dataset svhn --batch_size 256 --eps 0.0313 --attack_iters 100 \
    --out_datasets 'svhn'

