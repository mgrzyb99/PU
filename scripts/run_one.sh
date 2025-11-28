#!/bin/bash

#SBATCH --job-name PU
#SBATCH --partition short,experimental
#SBATCH --time 01:00:00
#SBATCH --array 0-9
#SBATCH --gpus 1
#SBATCH --mem 4G
#SBATCH --output .logs/%A/%A_%a.out
#SBATCH --error .logs/%A/%A_%a.out

shopt -s expand_aliases
source .venv/bin/activate

if [ $HOSTNAME = 'pascal' ]; then
    alias srun='/mnt/evafs/software/slurm/intel_broadwell/current/bin/srun'
else
    alias srun='/mnt/evafs/software/slurm/amd_epyc/current/bin/srun'
fi

srun python main.py fit \
    --config ${@: -1} ${@: 1:$#-1} \
    --trainer.enable_progress_bar false \
    --seed_everything $SLURM_ARRAY_TASK_ID
