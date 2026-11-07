#!/bin/bash
# Reproduce every number, table and figure in the paper, in order. Seeds are fixed inside each script.
# GPU jobs run one at a time (a 4 GB laptop GPU is enough). Probe/patching analyses that need no GPU run on the CPU.
# Total: roughly a day of compute. scripts/ops/ holds the queue scripts that were actually used.
set -e
cd "$(dirname "$0")/.."
TECTONIC=$(command -v tectonic || echo "$HOME/tools/tectonic/tectonic.exe")
CPU="env CUDA_VISIBLE_DEVICES= SUSS_DEV=cpu SUSS_THREADS=6"
mkdir -p runs results

# ---- Experiment A: in-context prediction of linear systems
for s in 0 1 2 3 4 5 6 7 8 9; do python scripts/train_a.py --seed $s; done   # 0-2 pre-registered, 3-9 replication
python scripts/eval_a.py --seeds 0 1 2 3 4 5 6 7 8 9
python scripts/mech_a.py                      # pre-registered probes / steering / patching / heads (seeds 0-2)
python scripts/mech_a2.py                     # exploratory: answer vs parameter probes
$CPU python scripts/mech_a3.py                # exploratory: query Jacobian, fast-weight probes
$CPU python scripts/mech_a4.py                # exploratory: KV-only patching (seeds 0-2)
$CPU SUSS_SEEDS=3,4,5,6,7,8,9 SUSS_OUT=results/a/mech4_rep.pt python scripts/mech_a4.py
$CPU python scripts/mech_a5.py                # exploratory: dimension-matched probe controls (seeds 0-2)
$CPU SUSS_SEEDS=3,4,5,6,7,8,9 SUSS_OUT=results/a/mech5_rep.pt python scripts/mech_a5.py
$CPU python scripts/mech_a6.py                # exploratory: shuffled labels, pooled residuals, ceiling, Remark check, top-k heads

# ---- Experiment B: MuJoCo pendulum, pushed block, bouncing ball
python scripts/gen_b.py
for sys in block pendulum bouncer; do for s in 0 1 2; do
  python scripts/train_b.py --system $sys --seed $s
  python scripts/train_b.py --system $sys --seed $s --mlp --steps 5000
done; done
python scripts/eval_b.py
$CPU python scripts/mech_b2.py                # key-value vs residual probes for physical parameters
SUSS_WORKERS=4 python scripts/sysid_block.py  # structured system identification with the exact simulator

# ---- Experiment C: in-context adaptive control
for s in 0 1 2; do
  python scripts/train_c.py --seed $s --fixed --device cpu
  python scripts/train_c2.py --seed $s --steps 6000                                                       # DAgger warm start
  python scripts/train_c.py --seed $s --steps 6000 --init runs/c/bc_seed$s.pt --loss lognorm --lr 1e-4 --tbptt 8   # cost fine-tuning
done
python scripts/eval_c.py
$CPU python scripts/mech_c.py                 # controller gain Jacobian, where B is stored
python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss raw --lr 1e-4 --tbptt 8 --tag _raw   # objective pilots
python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss cvar --lr 1e-4 --tbptt 8 --tag _cvar
mkdir -p results/pilots
for v in raw cvar; do grep -E "^step|^selected|^init" runs/c_tf_seed0_$v.log > results/pilots/final_warmstart_${v}_seed0.log || true; done

# ---- Scale study
python scripts/scale_a.py --tag long60k --steps 60000
python scripts/scale_a.py --tag deep8 --n_layer 8
python scripts/scale_a.py --tag wide256 --d_model 256
python scripts/scale_a.py --tag small64 --d_model 64 --n_layer 2
for s in 0 1 2; do python scripts/scale_a.py --tag n4m2_s$s --n 4 --m 2 --seed $s; done

# ---- Pre-registration 2 (PREREGISTRATION_2.md, tag prereg2-94b6cf8): confirmatory tests, CPU
$CPU python scripts/confirm_a.py                # KV account on long60k, deep8, wide256 (+ small64)
$CPU python scripts/confirm_b.py                # prior shift: query Jacobian vs ridge (training / new prior) and OLS
python scripts/confirm_stats.py                 # the 11 registered tests, Holm

# ---- Statistics, numbers, figures, tables, paper, packages
python scripts/analyze.py
python scripts/make_numbers.py
python scripts/make_figures.py
python scripts/make_web_figures.py
python scripts/pilot_table.py
python scripts/seed_table.py
python scripts/scale_table.py
python scripts/controls_table.py
python scripts/confirm_table.py
python scripts/check_macros.py
(cd paper && "$TECTONIC" -X compile --keep-intermediates main.tex)
bash scripts/package.sh
python scripts/pack_models.py
python scripts/build_notebook.py              # re-executes notebooks/quickstart.ipynb (CPU, a few minutes)
python scripts/export_demo.py                 # weights and reference curves for the in-browser demo (docs/demo/)
