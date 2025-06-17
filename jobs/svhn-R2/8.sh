#!/bin/bash

#SBATCH --account=sunwbgt0
#SBATCH --job-name=j8
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/ATD/out/svhn-R2-Imbalanced/8.log

python train_ATD.py  --run_name svhn-R2-Imbalanced-8 --model_type "fea" --training_type "adv" \
        --in_dataset svhn-R2 --alpha 0.5 --batch_size 256 --num_epochs 20 --eps 0.0313 --attack_iters 10 \
        --regime Imbalanced --ood_dset svhn-R2 --n_ood 8 --valsize 10000
python test_ATD.py  --run_name svhn-R2-Imbalanced-8 --model_type "fea" --in_dataset svhn-R2 --batch_size 256 --eps 0.0313 --attack_iters 100 \
            --out_datasets svhn-R2
