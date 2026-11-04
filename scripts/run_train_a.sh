#!/bin/bash
cd "$(dirname "$0")/.."
for s in 0 1 2; do python scripts/train_a.py --seed $s > runs/a_train_seed$s.log 2>&1; done
