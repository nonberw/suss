#!/bin/bash
# Controller queue (runs in parallel with queue_s.sh while SUSS has the GPU to itself).
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
# wait for any controller-training process that queue6 started (it keeps running after queue6 is stopped)
while powershell -c "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -match 'train_c2?\.py' } | Select-Object -ExpandProperty CommandLine" | grep -q train_c; do sleep 30; done
for s in 1 2; do
  run runs/c/bc_seed$s.pt runs/c_bc_seed$s.log python scripts/train_c2.py --seed $s --steps 6000
  run runs/c/seed$s.pt runs/c_tf_seed$s.log python scripts/train_c.py --seed $s --steps 6000 --init runs/c/bc_seed$s.pt --loss lognorm --lr 1e-4 --tbptt 8
done
run results/c/eval.pt runs/eval_c.log python scripts/eval_c.py
run results/c/mech.pt runs/mech_c.log python scripts/mech_c.py
echo QUEUE_C_DONE
