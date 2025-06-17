import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="details")
parser.add_argument('--regime', type=str)
parser.add_argument('--experiment', type=str)
args = parser.parse_args()

def read_last_auc(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()[-4:]  # Get last 4 lines

    for line in lines:
        if 'Test AUC' in line:
            # Extract the numeric part before the '%' sign
            try:
                percent_str = line.split('Test AUC:')[1].split('%')[0].strip()
                return float(percent_str)
            except (IndexError, ValueError):
                pass  # Malformed line, skip or raise error if needed

    return None  # AUC not found

def read_last_tpr95(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()[::-1]  # Read lines in reverse order

    for line in lines:
        if 'Testing TPR95' in line:
            try:
                return float(line.split('Testing TPR95:')[1].strip())
            except (IndexError, ValueError):
                pass

    return None  # TPR95 not found

def read_last_tpr99(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()[::-1]  # Read lines in reverse order

    for line in lines:
        if 'Testing TPR99' in line:
            try:
                return float(line.split('Testing TPR99:')[1].strip())
            except (IndexError, ValueError):
                pass

    return None  # TPR95 not found

EXP_DSET = args.experiment
regime = args.regime
N = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# N = [8, 16, 32]

AUCs = []
TPR95 = []
TPR99 = []
for n in N:
    file_path = os.path.join('..', 'log', 'DeepSAD', f'{EXP_DSET}-{regime}-{n}', 'log.txt')
    auc_value = read_last_auc(file_path)
    # print(f"Parsed AUC: {auc_value}%")
    # print(f"Numeric AUC: {auc_value}")
    tpr95 = read_last_tpr95(file_path)
    # print(f"TPR95: {tpr95}")

    tpr99 = read_last_tpr99(file_path)
    # print(f"TPR95: {tpr99}")

    AUCs.append(auc_value)
    TPR95.append(tpr95 * 100)
    TPR99.append(tpr99 * 100)


print(f"Summary for {EXP_DSET} with {regime} regime:")
print(f"N: {N}")
print(f"AUCs: {', '.join(f'{f:.4f}' for f in AUCs)}")
print(f"TPR95: {', '.join(f'{f:.4f}' for f in TPR95)}")
print(f"TPR99: {', '.join(f'{f:.4f}' for f in TPR99)}\n\n")