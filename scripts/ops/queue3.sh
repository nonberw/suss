#!/bin/bash
# Robust sequential queue: one GPU job at a time, skip finished outputs, retry on CUDA failures.
cd "$(dirname "$0")/.."
run() {  # run <output-file> <log> <cmd...>
  out=$1; log=$2; shift 2
  for try in 1 2 3; do
    [ -f "$out" ] && return 0
    echo "$(date +%T) start $* (try $try)"
    "$@" > "$log" 2>&1
    [ -f "$out" ] && { echo "$(date +%T) done $out"; return 0; }
    echo "$(date +%T) FAILED $out (try $try)"; sleep 60
  done
}
for s in 0 1 2; do
  run runs/b/pendulum_tf_seed$s.pt runs/b_pendulum_tf_seed$s.log python scripts/train_b.py --system pendulum --seed $s
  run runs/b/pendulum_mlp_seed$s.pt runs/b_pendulum_mlp_seed$s.log python scripts/train_b.py --system pendulum --seed $s --mlp --steps 5000
done
run results/b/pendulum_mech.pt runs/eval_b.log python scripts/eval_b.py
run results/a/mech2.pt runs/mech_a2.log python scripts/mech_a2.py
for s in 0 1 2; do
  run runs/c/seed$s.pt runs/c_tf_seed$s.log python scripts/train_c.py --seed $s
done
run results/c/eval.pt runs/eval_c.log python scripts/eval_c.py
echo QUEUE3_DONE
