#!/bin/bash

shopt -s extglob globstar

for config; do
    sbatch scripts/run_one.sh $config
done
