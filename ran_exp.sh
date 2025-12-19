#!/bin/bash
python scripts/train.py --method tl1 --lr0 0.0003 --data datasets/generated/mixed_10pct_random.yaml
python scripts/train.py --method tl1 --lr0 0.0003 --data datasets/generated/mixed_10pct_stratified.yaml
python scripts/train.py --method tl2 --lr0 0.0003 --freeze 10 --data datasets/generated/mixed_10pct_random.yaml
python scripts/train.py --method tl2 --lr0 0.0003 --freeze 22 --data datasets/generated/mixed_10pct_random.yaml
python scripts/train.py --method tl2 --lr0 0.0003 --freeze 10 --data datasets/generated/mixed_10pct_stratified.yaml
python scripts/train.py --method tl2 --lr0 0.0003 --freeze 22 --data datasets/generated/mixed_10pct_stratified.yaml