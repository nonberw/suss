#!/bin/bash
# Reproduce every number in the paper. Seeds are fixed inside each script. One GPU job at a time.
set -e
cd "$(dirname "$0")/.."
mkdir -p runs results
# ---- Experiment A (pre-registered) + exploratory mechanism
for s in 0 1 2; do python scripts/train_a.py --seed $s; done
python scripts/eval_a.py
python scripts/mech_a.py      # pre-registered probes / steering / patching / heads
python scripts/mech_a2.py     # exploratory: answer vs parameter probes
python scripts/mech_a3.py     # exploratory: query Jacobian, fast-weight probes
python scripts/mech_a4.py     # exploratory: KV-only patching
# ---- Experiment B
python scripts/gen_b.py
for sys in block pendulum bouncer; do for s in 0 1 2; do
  python scripts/train_b.py --system $sys --seed $s
  python scripts/train_b.py --system $sys --seed $s --mlp --steps 5000
done; done
python scripts/eval_b.py
python scripts/mech_b2.py
# ---- Experiment C
for s in 0 1 2; do
  python scripts/train_c.py --seed $s --fixed --device cpu
  python scripts/train_c2.py --seed $s --steps 6000                                   # stage 1: DAgger warm start
  python scripts/train_c.py --seed $s --steps 6000 --init runs/c/bc_seed$s.pt --loss lognorm --lr 1e-4 --tbptt 8  # stage 2
done
python scripts/eval_c.py
python scripts/mech_c.py     # exploratory: controller gain Jacobian, where B is stored
# ---- Scale study (exploratory)
python scripts/scale_a.py --tag long60k --steps 60000
for s in 0 1 2; do python scripts/scale_a.py --tag n4m2_s$s --n 4 --m 2 --seed $s; done
python scripts/scale_a.py --tag deep8 --n_layer 8
python scripts/scale_a.py --tag wide256 --d_model 256
python scripts/scale_a.py --tag small64 --d_model 64 --n_layer 2
# ---- Statistics, numbers, figures, paper
python scripts/analyze.py
python scripts/make_numbers.py
python scripts/make_figures.py
python scripts/pilot_table.py   # pilots (seed 99) logged in results/pilots/
(cd paper && tectonic -X compile --keep-intermediates main.tex)
bash scripts/package.sh
