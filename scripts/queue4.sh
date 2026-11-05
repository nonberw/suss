#!/bin/bash
# Exploratory extensions, strictly after queue3 (one GPU job at a time).
cd "$(dirname "$0")/.."
until grep -q QUEUE3_DONE runs/queue3.log 2>/dev/null; do sleep 30; done
until [ -f data/bouncer_train.npz ]; do sleep 30; done
run() {
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
  run runs/b/bouncer_tf_seed$s.pt runs/b_bouncer_tf_seed$s.log python scripts/train_b.py --system bouncer --seed $s
  run runs/b/bouncer_mlp_seed$s.pt runs/b_bouncer_mlp_seed$s.log python scripts/train_b.py --system bouncer --seed $s --mlp --steps 5000
done
rm -f results/b/bouncer_mech.pt
run results/b/bouncer_mech.pt runs/eval_b2.log python scripts/eval_b.py
run results/scale/long60k.json runs/scale_long60k.log python scripts/scale_a.py --tag long60k --steps 60000
for s in 0 1 2; do
  run results/scale/n4m2_s$s.json runs/scale_n4m2_s$s.log python scripts/scale_a.py --tag n4m2_s$s --n 4 --m 2 --seed $s
done
run results/scale/deep8.json runs/scale_deep8.log python scripts/scale_a.py --tag deep8 --n_layer 8
run results/scale/wide256.json runs/scale_wide256.log python scripts/scale_a.py --tag wide256 --d_model 256
run results/scale/small64.json runs/scale_small64.log python scripts/scale_a.py --tag small64 --d_model 64 --n_layer 2
run results/scale/n4m2_long60k.json runs/scale_n4m2_long.log python scripts/scale_a.py --tag n4m2_long60k --n 4 --m 2 --steps 60000
echo QUEUE4_DONE
