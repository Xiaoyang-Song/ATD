import os
import numpy as np


# GL
ACCOUNT = 'sunwbgt0'
TIME = "4:00:00"
# Configuration
# EXP_DSET = 'fashionmnist'
# EXP_DSET = 'fashionmnist-R2'
# EXP_DSET = 'cifar10-svhn'
# EXP_DSET = 'mnist'
# EXP_DSET = 'mnist-fashionmnist' 
# EXP_DSET = 'svhn' 
EXP_DSET = 'svhn-R2'
regime = 'Imbalanced'
# regime = 'Balanced'


N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

CMD_ONLY = False


for n in N:
    print(f"sbatch jobs/{EXP_DSET}/{n}.sh")

if not CMD_ONLY:
    print("Generating job files...")
    # Create logging directory
    log_path = os.path.join('checkpoint', 'log', EXP_DSET)
    os.makedirs(log_path, exist_ok=True)

    for n in N:
        # Create job directory
        job_path = os.path.join('jobs', EXP_DSET)
        os.makedirs(job_path, exist_ok=True)
        # Declare job name
        filename = os.path.join('jobs', EXP_DSET, f"{n}.sh")
        # Write files
        f = open(filename, 'w')
        f.write("#!/bin/bash\n\n")
        f.write(f"#SBATCH --account={ACCOUNT}\n")
        f.write(f"#SBATCH --job-name=j{n}\n")
        f.write("#SBATCH --mail-user=xysong@umich.edu\n")
        f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --partition=gpu\n")
        f.write("#SBATCH --gpus=1\n")
        f.write("#SBATCH --mem-per-gpu=16GB\n")
        f.write(f"#SBATCH --time={TIME}\n")
        f.write(f"#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/ATD/out/{EXP_DSET}-{regime}/{n}.log\n\n")
        
        f.write(
            f"""python train_ATD.py  --run_name {EXP_DSET} --model_type "fea" --training_type "adv" \\
        --in_dataset {EXP_DSET} --alpha 0.5 --batch_size 256 --num_epochs 20 --eps 0.0313 --attack_iters 10 \\
        --regime {regime} --ood_dset {EXP_DSET} --n_ood {n} --valsize 10000\n"""
        )

        f.write(
            f"""python test_ATD.py  --run_name {EXP_DSET} --model_type "fea" --in_dataset {EXP_DSET} --batch_size 256 --eps 0.0313 --attack_iters 100 \\
            --out_datasets {EXP_DSET}\n"""
        )

        f.close()