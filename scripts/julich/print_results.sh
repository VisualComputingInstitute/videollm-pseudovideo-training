#!/bin/bash -x

source scripts/julich/setup_env.sh

experiment_name=${1}
base_eval_dir=${experiments_dir}/${experiment_name}/eval_results
experiment_dir=${experiments_dir}/${experiment_name}
if [ ! -d $experiment_dir ]; then
    echo "\"$experiment_name\" does not appear to be an existing experiment"
    exit -1
fi

pad=$(printf '%0.1s' "-"{1..120})
leftpadlength=20
rightpadlength=100

IFS=',' read -r -a cmd_override_datasets <<< "$2"
if [ ${#cmd_override_datasets[@]} -ne 0 ]; then
    known_tvbench_style_datasets=("${cmd_override_datasets[@]}")
fi
for dataset in "${known_tvbench_style_datasets[@]}"; do
    if [ "${dataset}" = "tvbench" ]; then
        eval_modes=("motion_eval_simple" "motion_eval_complete" "tvbench_original_eval")
    else
        eval_modes=("motion_eval_simple" "motion_eval_complete")
    fi
    for mode in "${eval_modes[@]}"; do
        print_str="Results for ${dataset}, eval mode \"${mode}\""
        acc_file=${base_eval_dir}/${dataset}_${mode}/acc_results.json
        printf '%*.*s' 0 $leftpadlength "$pad"
        printf "$print_str"
        printf '%*.*s' 0 $((rightpadlength - ${#print_str})) "$pad"
        if [ -f ${acc_file} ]; then
            printf "\n\n"
            cat $acc_file 
            printf "\nPresumably last evaluated on "
            date -r $acc_file +"%Y-%m-%d %H:%M:%S"
            printf "\n"
        else
            printf "\n\nProbably not evaluated yet\n\n"
        fi
    done
done