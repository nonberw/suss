#!/bin/bash
cd "$(dirname "$0")/.."
until [ -f results/b/pendulum_mech.pt ] || grep -q Traceback runs/eval_b.log 2>/dev/null; do sleep 30; done
python scripts/mech_a2.py > runs/mech_a2.log 2>&1
for s in 0 1 2; do python scripts/train_c.py --seed $s > runs/c_tf_seed$s.log 2>&1; done
python scripts/eval_c.py > runs/eval_c.log 2>&1
echo QUEUE2_DONE
