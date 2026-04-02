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

#if you don't want to schedule a job for each dataset, provide desired
#datasets via the 2nd command line argument as a comma-separated string
IFS=',' read -r -a cmd_override_datasets <<< "$2"
if [ ${#cmd_override_datasets[@]} -ne 0 ]; then
    known_tvbench_style_datasets=("${cmd_override_datasets[@]}")
fi

declare -A dataset_locations
dataset_locations["TVBenchMoveDirShuffled"]="${PROJECT}/workspaces/\
videollava_workspace/eval/TVBench"
dataset_locations["Linear2kShuffled"]="${longvu_workspace}//val_data/lin2k"
dataset_locations["Linear4dirs"]="${longvu_workspace}/val_data/lin2k"
dataset_locations["Linear2k"]="${longvu_workspace}/val_data/lin2k"
dataset_locations["TVBench"]="${PROJECT}/workspaces/videollava_workspace/\
eval/TVBench"
dataset_locations["MoreThanOneObj"]="${longvu_workspace}/val_data/\
more_than_one_obj"

declare -A dataset_runtimes
dataset_runtimes["TVBenchMoveDirShuffled"]="00:15:00"
dataset_runtimes["Linear2kShuffled"]="00:15:00"
dataset_runtimes["Linear4dirs"]="00:15:00"
dataset_runtimes["Linear2k"]="00:15:00"
dataset_runtimes["TVBench"]="00:45:00"
dataset_runtimes["MoreThanOneObj"]="00:15:00"

update_codebase="1"
if [ "$update_codebase" -eq 1 ]; then
    rsync -az --exclude=.git --exclude=examples . ${experiment_dir}/code
fi

for dataset in "${known_tvbench_style_datasets[@]}"; do
    slurm_dir=${experiment_dir}/slurm
    dataset_loc=${dataset_locations[$dataset]}
    log_file=${slurm_dir}/slurm_eval_${dataset}.txt
    sbatch --job-name="${experiment_name}_eval_${dataset}" --output=$log_file \
        --time=${dataset_runtimes[$dataset]}  \
        scripts/julich/sbatch_tvbench_format.sh $experiment_dir $dataset \
        $dataset_loc
done