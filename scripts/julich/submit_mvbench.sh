#!/bin/bash -x

set -e
source scripts/julich/setup_env.sh

experiment_name=${1}
experiment_dir=${experiments_dir}/${experiment_name}
if [ ! -d $experiment_dir ]; then
    echo "\"$experiment_name\" does not appear to be an existing experiment"
    exit -1
fi
flag_file="${experiment_dir}/slurm/training_complete.flag"
if [ ! -f $flag_file ]; then
    echo "\"$experiment_name\" does not appear to have finished (training_complete.flag not found)"
    exit -1
fi

update_codebase="0"
if [ "$update_codebase" -eq 1 ]; then
    rsync -az ./longvu/ ${experiment_dir}/code/longvu
    rsync -az ./eval/ ${experiment_dir}/code/eval
fi

slurm_dir=${experiment_dir}/slurm
log_file=${slurm_dir}/slurm_eval_mvbench.txt
dataset_loc=${PROJECT}/workspaces/datasets/MVBench
sbatch --job-name="${experiment_name}_eval_mvbench" \
    --output=$log_file --time="00:20:00" \
    scripts/julich/sbatch_mvbench.sh $experiment_dir \
    $dataset_loc "$@"