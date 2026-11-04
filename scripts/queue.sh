#!/bin/bash
cd "$(dirname "$0")/.."
until [ -f runs/a/seed2.pt ] && [ -f runs/c/fixed_seed2.pt ]; do sleep 10; done
bash scripts/run_c.sh &
bash scripts/run_rest.sh
wait
echo QUEUE_DONE
