#!/bin/bash

shopt -s extglob globstar

for config; do
    for positive_labels in 0 1 8 9; do
        sbatch scripts/run_one.sh --data.init_args.pn_wrap_kwargs.positive_labels $positive_labels $config
    done
done
