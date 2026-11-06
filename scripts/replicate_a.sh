#!/bin/bash
# Exploratory replication of Experiment A on 7 additional training seeds (3..9), seed as the unit of analysis.
# GPU: eval only (one job). CPU (capped threads, no CUDA context): KV-only patching and dimension-matched probes.
cd "$(dirname "$0")/.."
until [ -f runs/a/seed9.pt ]; do sleep 30; done
python scripts/eval_a.py --seeds 3 4 5 6 7 8 9 > runs/eval_a_rep.log 2>&1
CUDA_VISIBLE_DEVICES= SUSS_DEV=cpu SUSS_THREADS=6 SUSS_SEEDS=3,4,5,6,7,8,9 SUSS_OUT=results/a/mech4_rep.pt python scripts/mech_a4.py > runs/mech_a4_rep.log 2>&1
CUDA_VISIBLE_DEVICES= SUSS_DEV=cpu SUSS_THREADS=6 SUSS_SEEDS=3,4,5,6,7,8,9 SUSS_OUT=results/a/mech5_rep.pt python scripts/mech_a5.py > runs/mech_a5_rep.log 2>&1
echo REPLICATE_DONE
