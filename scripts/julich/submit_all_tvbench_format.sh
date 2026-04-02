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
#if you want to provide extra arguments to the tvbench python script, provide
#these as the third argument, and ensure that you give "" as the second
IFS=',' read -r -a cmd_override_datasets <<< "$2"
if [ ${#cmd_override_datasets[@]} -ne 0 ]; then
    known_tvbench_style_datasets=("${cmd_override_datasets[@]}")
    shift 2
else
    shift 1
fi

declare -A dataset_locations
dataset_locations["tvbench"]="${PROJECT}/workspaces/videollava_workspace/\
eval/TVBench"
dataset_locations["one_obj_lin"]="${longvu_workspace}/val_data/one_obj_lin"
dataset_locations["one_obj_lin_buggy"]="${longvu_workspace}/val_data/\
one_obj_lin_buggy"
dataset_locations["more_objs_lin"]="${longvu_workspace}/val_data/more_objs_lin"
dataset_locations["four_new_colors"]="${longvu_workspace}/val_data/\
four_new_colors"
dataset_locations["more_objs_lin_count"]="${longvu_workspace}/val_data/more_objs_lin_count"

declare -A runtimes_simple
runtimes_simple["one_obj_lin"]="00:15:00"
runtimes_simple["one_obj_lin_buggy"]="00:15:00"
runtimes_simple["tvbench"]="00:15:00"
runtimes_simple["more_objs_lin"]="00:15:00"
runtimes_simple["four_new_colors"]="00:15:00"

declare -A runtimes_complete
runtimes_complete["one_obj_lin_buggy"]="01:15:00"
runtimes_complete["one_obj_lin"]="01:15:00"
runtimes_complete["tvbench"]="00:45:00"
runtimes_complete["more_objs_lin"]="01:45:00"
runtimes_complete["four_new_colors"]="01:15:00"

update_codebase="1"
if [ "$update_codebase" -eq 1 ]; then
    rsync -az ./longvu/ ${experiment_dir}/code/longvu
    rsync -az ./eval/ ${experiment_dir}/code/eval
fi

slurm_dir=${experiment_dir}/slurm
for dataset in "${known_tvbench_style_datasets[@]}"; do
    
    dataset_loc=${dataset_locations[$dataset]}
    if [ "${dataset}" = "more_objs_lin_count" ]; then
        log_file=${slurm_dir}/slurm_eval_more_objs_lin_count.txt
        sbatch --job-name="${experiment_name}_eval_more_objs_lin_count" \
            --output=$log_file --time="00:15:00" \
            scripts/julich/sbatch_tvbench_format.sh $experiment_dir $dataset \
            "counting" $dataset_loc "$@"
    else
        #for TVBench, dispatch a job for the original dataset too
        if [ "${dataset}" = "tvbench" ]; then
            log_file=${slurm_dir}/slurm_eval_${dataset}_full_original.txt
            sbatch --job-name="${experiment_name}_eval_${dataset}_full_original" \
                --output=$log_file --time="00:45:00" \
                scripts/julich/sbatch_tvbench_format.sh $experiment_dir $dataset \
                "tvbench_original_eval" $dataset_loc "$@"
        fi

        log_file=${slurm_dir}/slurm_eval_${dataset}_simple.txt
        sbatch --job-name="${experiment_name}_eval_${dataset}_simple" \
            --output=$log_file --time=${runtimes_simple[$dataset]}  \
            scripts/julich/sbatch_tvbench_format.sh $experiment_dir $dataset \
            "motion_eval_simple" $dataset_loc "$@"

        log_file=${slurm_dir}/slurm_eval_${dataset}_complete.txt
        sbatch --job-name="${experiment_name}_eval_${dataset}_complete" \
            --output=$log_file --time=${runtimes_complete[$dataset]}  \
            scripts/julich/sbatch_tvbench_format.sh $experiment_dir $dataset \
            "motion_eval_complete" $dataset_loc "$@"
    fi
done
