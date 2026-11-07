#!/bin/bash
# GPU queue after the review, part 2 (started once the controller pilots of queue7 are running; queue7's bash is
# stopped so that it does not start the scale runs). One GPU job at a time. Priority: the two most informative scale
# runs, then 7 extra seeds for Experiment A (10 seeds in total for the mechanism replication), then the rest.
cd "$(dirname "$0")/.."
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
# wait for any controller-pilot training started by queue7
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'train_c.py' } | Select-Object -ExpandProperty CommandLine" | grep -q train_c; do sleep 30; done
run runs/c/seed0_raw.pt runs/c_tf_seed0_raw.log python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss raw --lr 1e-4 --tbptt 8 --tag _raw
run runs/c/seed0_cvar.pt runs/c_tf_seed0_cvar.log python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss cvar --lr 1e-4 --tbptt 8 --tag _cvar
run results/scale/long60k.json runs/scale_long60k.log python scripts/scale_a.py --tag long60k --steps 60000
run results/scale/n4m2_s0.json runs/scale_n4m2_s0.log python scripts/scale_a.py --tag n4m2_s0 --n 4 --m 2 --seed 0
for s in 3 4 5 6 7 8 9; do
  run runs/a/seed$s.pt runs/a_train_seed$s.log python scripts/train_a.py --seed $s
done
run results/scale/deep8.json runs/scale_deep8.log python scripts/scale_a.py --tag deep8 --n_layer 8
run results/scale/wide256.json runs/scale_wide256.log python scripts/scale_a.py --tag wide256 --d_model 256
run results/scale/small64.json runs/scale_small64.log python scripts/scale_a.py --tag small64 --d_model 64 --n_layer 2
for s in 1 2; do
  run results/scale/n4m2_s$s.json runs/scale_n4m2_s$s.log python scripts/scale_a.py --tag n4m2_s$s --n 4 --m 2 --seed $s
done
echo QUEUE8_DONE
