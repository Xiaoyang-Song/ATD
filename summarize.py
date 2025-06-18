import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="details")
parser.add_argument('--regime', type=str)
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

EXP_DSET = args.experiment
regime = args.regime
N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# N = [16] # debug usage

AUCs = []
TPR95 = []
TPR99 = []
for n in N:
    file_path = os.path.join('out', f'{EXP_DSET}-{regime}', f'{n}.log')
    with open(file_path, 'r') as f:
            lines = f.readlines()
            # print(f'{EXP_DSET}-{regime}', n)
            auc = float(lines[-12].strip()[10:])
            tpr95 = float(lines[-11].strip()[15:])
            tpr99 = float(lines[-10].strip()[15:])
            # print(auc, tpr95, tpr99)
            AUCs.append(auc * 100)
            TPR95.append(tpr95 * 100)
            TPR99.append(tpr99 * 100)

assert len(AUCs) == len(N) == len(TPR95) == len(TPR99), f"{EXP_DSET} {regime} has different lengths: {len(AUCs)}, {len(N)}, {len(TPR95)}, {len(TPR99)}"

print(f"Summary for {EXP_DSET} with {regime} regime:")
print(f"N: {N}")
print(f"AUCs: {', '.join(f'{f:.4f}' for f in AUCs)}")
print(f"TPR95: {', '.join(f'{f:.4f}' for f in TPR95)}")
print(f"TPR99: {', '.join(f'{f:.4f}' for f in TPR99)}\n\n")