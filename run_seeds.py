import os
import sys

# 반복할 시드 범위
#seeds = [1834, 3721, 2829, 3049, 5731, 5729, 2194, 4910, 5810, 942]
#seeds = [4821, 9372, 161, 7059, 2880, 6573, 894, 2134, 7596, 3741]
seeds = [2411, 5193, 4594]
dataset = "waterbirds"
#dataset = "celeba"

for seed in seeds:
    print(f"\n===== [Seed {seed}] Training Baseline =====")
    os.system(f"{sys.executable} margin_loss.py --dataset {dataset} --train --type baseline --bias --seed {seed}")

    print(f"\n===== [Seed {seed}] Training Margin =====")
    os.system(f"{sys.executable} margin_loss.py --dataset {dataset} --train --type margin --seed {seed}")

    #print(f"\n===== [Seed {seed}] Testing Baseline =====")
    #os.system(f"{sys.executable} margin_loss.py --dataset {dataset} --test-only --type baseline --seed {seed}")

    print(f"\n===== [Seed {seed}] Testing Margin =====")
    os.system(f"{sys.executable} margin_loss.py --dataset {dataset} --test-only --type margin --seed {seed}")