#!/bin/bash
# Reproduce every number in the paper. Seeds are fixed inside each script.
set -e
cd "$(dirname "$0")/.."
mkdir -p runs results
for s in 0 1 2; do python scripts/train_a.py --seed $s; done
python scripts/eval_a.py
python scripts/mech_a.py
python scripts/mech_a2.py
python scripts/gen_b.py
for sys in block pendulum; do for s in 0 1 2; do
  python scripts/train_b.py --system $sys --seed $s
  python scripts/train_b.py --system $sys --seed $s --mlp --steps 5000
done; done
python scripts/eval_b.py
for s in 0 1 2; do
  python scripts/train_c.py --seed $s --fixed --device cpu
  python scripts/train_c.py --seed $s
done
python scripts/eval_c.py
python scripts/analyze.py
python scripts/make_figures.py
(cd paper && tectonic main.tex)
