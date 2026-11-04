#!/bin/bash
# Full pipeline after Experiment A training. Each step logs to runs/*.log
cd "$(dirname "$0")/.."
set -x
python scripts/eval_a.py > runs/eval_a.log 2>&1
python scripts/mech_a.py > runs/mech_a.log 2>&1
for sys in block pendulum; do for s in 0 1 2; do
  python scripts/train_b.py --system $sys --seed $s > runs/b_${sys}_tf_seed$s.log 2>&1
  python scripts/train_b.py --system $sys --seed $s --mlp --steps 5000 > runs/b_${sys}_mlp_seed$s.log 2>&1
done; done
python scripts/eval_b.py > runs/eval_b.log 2>&1
