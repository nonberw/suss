#!/bin/bash
cd "$(dirname "$0")/.."
for s in 0 1 2; do python scripts/train_c.py --seed $s > runs/c_tf_seed$s.log 2>&1; done
