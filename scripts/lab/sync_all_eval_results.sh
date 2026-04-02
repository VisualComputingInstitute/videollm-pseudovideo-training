#!/bin/bash -x

experiment_name=$1
last_N_parts_to_use_for_name=$2
remote_base_exp_dir=/p/project1/efficientvid/workspaces/longvu_workspace/experiments
local_base_exp_dir=/work/lydakis/longvu_workspace/experiments

mkdir -p ${local_base_exp_dir}/${experiment_name}
rsync -avz juwels-booster:${remote_base_exp_dir}/${experiment_name}/eval_results \
    ${local_base_exp_dir}/${experiment_name}
rsync -avz juwels-booster:${remote_base_exp_dir}/${experiment_name}/training_logs \
    ${local_base_exp_dir}/${experiment_name}    

if [ "$last_N_parts_to_use_for_name" -eq 0 ]; then
    echo "Skipping report creation"
else
    python ~/work/longvu/result_analysis/create_model_report.py \
        ${local_base_exp_dir}/${experiment_name} $last_N_parts_to_use_for_name
fi