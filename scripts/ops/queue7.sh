#!/bin/bash
# GPU queue after the review: controller objective pilots before the scale study. One GPU job at a time.
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
# wait for the bouncer seed-2 training that queue_s started
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'train_b.py --system bouncer' } | Select-Object -ExpandProperty CommandLine" | grep -q train_b; do sleep 30; done
run runs/b/bouncer_mlp_seed2.pt runs/b_bouncer_mlp_seed2.log python scripts/train_b.py --system bouncer --seed 2 --mlp --steps 5000
run results/b/bouncer_mech.pt runs/eval_b2.log python scripts/eval_b.py
rm -f results/b/mech2.pt
run results/b/mech2.pt runs/mech_b2.log python scripts/mech_b2.py
# controller objective pilots (warm start + raw cost, warm start + CVaR), seed 0, 3000 steps each
run runs/c/seed0_raw.pt runs/c_tf_seed0_raw.log python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss raw --lr 1e-4 --tbptt 8 --tag _raw
run runs/c/seed0_cvar.pt runs/c_tf_seed0_cvar.log python scripts/train_c.py --seed 0 --steps 3000 --init runs/c/bc_seed0.pt --loss cvar --lr 1e-4 --tbptt 8 --tag _cvar
# scale study
run results/scale/long60k.json runs/scale_long60k.log python scripts/scale_a.py --tag long60k --steps 60000
run results/scale/n4m2_s0.json runs/scale_n4m2_s0.log python scripts/scale_a.py --tag n4m2_s0 --n 4 --m 2 --seed 0
run results/scale/deep8.json runs/scale_deep8.log python scripts/scale_a.py --tag deep8 --n_layer 8
run results/scale/wide256.json runs/scale_wide256.log python scripts/scale_a.py --tag wide256 --d_model 256
run results/scale/small64.json runs/scale_small64.log python scripts/scale_a.py --tag small64 --d_model 64 --n_layer 2
for s in 1 2; do
  run results/scale/n4m2_s$s.json runs/scale_n4m2_s$s.log python scripts/scale_a.py --tag n4m2_s$s --n 4 --m 2 --seed $s
done
echo QUEUE7_DONE
