#!/bin/bash

shopt -s extglob globstar

for config; do
    for mixup_gamma in 0.1 0.3 1.0 3.0; do
        sbatch scripts/run_one.sh --model.init_args.mixup_gamma $mixup_gamma $config
    done
done
