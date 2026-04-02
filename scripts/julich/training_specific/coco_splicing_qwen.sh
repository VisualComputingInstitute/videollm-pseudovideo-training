#!/bin/bash -x

set -e
#This script expects to be executed with cwd=longvu/
#NOTE: this sets several environment variables used below
source scripts/julich/setup_env.sh

experiment_name=$1
shift 1
date_time=$(date +"%Y_%m_%d_%Hh_%Mm_%Ss")
experiment_dir=${experiments_dir}/${experiment_name}_${date_time}
slurm_dir=${experiment_dir}/slurm
ckpt_dir=${experiment_dir}/checkpoints
logging_dir=${experiment_dir}/training_logs
mkdir -p $experiment_dir
mkdir -p $slurm_dir
mkdir -p $ckpt_dir
mkdir -p $logging_dir

copy_code_to=${experiment_dir}/code
echo "Code will be copied to " $copy_code_to
rsync -az --exclude=.git --exclude=examples . $copy_code_to
echo "Code copied succesfully"
copied_batch_file=${copy_code_to}/scripts/julich/sbatch_train_zero2_offl.sh

#NOTE: backslashes are escaped so they'll be preserved upon copying in the
#actual sbatch file
training_args=("--gradient_checkpointing" "True" \
"--max_steps" "312" \
"--per_device_train_batch_size" "1" \
"--per_device_eval_batch_size" "4" \
"--gradient_accumulation_steps" "2" \
"--eval_steps" "500" \
"--evaluation_strategy" "no" \
"--learning_rate" "5e-6" \
"--weight_decay" "0." \
"--warmup_ratio" "0.03" \
"--lr_scheduler_type" "cosine" \
"--tf32" "False" \
"--fp16" "False" \
"--bf16" "True" \
"--dataloader_num_workers" "0" \
"--group_by_modality_length" "True" \
"--lazy_preprocess" "True" \
"--logging_steps" "10" \
"--log_on_each_node" "False" \
"--save_total_limit" "3" \
"--save_steps" "300" \
"--save_strategy" "steps" \
"--report_to" "tensorboard" \
"--resume_from_checkpoint" "True" \
"--seed" "42")

#input_model_filename refers to where to finetune from, and should contain
#"cambrian" somewhere in its filepath (at least with the current code version)
#version is either llama3 or qwen
version="qwen"
#we use this later for instantiation during evals
echo $version >> $experiment_dir/version_file.txt

filesystem_related_args=("--output_dir" "${ckpt_dir}" \
"--version" "${version}" \
"--input_model_filename" \
"./pretr_checkpoints/cambrian/LongVU_Qwen2_7B_video_ft" \
"--output_model_filename" "${ckpt_dir}" \
"--slurm_dir" "${slurm_dir}" \
"--logging_dir" "${logging_dir}")

#it gets a bit complicated with backslashes and quotes
s="\\\\"
s2="\\\\\\\\"
s3="\\\\\\\\\\\\"
data_args=("--dataset_type" "coco_img_splicing" \
"--dataset_kwargs_as_dict_str" \
"{${s}\"coco_yaml_path${s}\": \
${s}\"./data_configs/splicing/coco_captioning_julich.yaml${s}\", \
${s}\"question_yaml_path${s}\": \
${s}\"./data_configs/splicing/sc_descr_first_last_qwen.yaml${s}\"}")

llm_and_freezing_args=("--model_max_length" "8192" \
"--mm_vision_select_layer" "-2" \
"--mm_use_im_start_end" "False" \
"--mm_use_im_patch_token" "False" \
"--image_aspect_ratio" "pad" \
"--tune_mm_mlp_adapter" "False" \
"--freeze_mm_mlp_adapter" "False" \
"--freeze_backbone" "False" \
"--mm_projector_type" "sva")

#video_fps: how many frames to sample per second
longvu_algorithm_args=("--query_num_list" "[144]" \
"--image_token_len" "144" \
"--lowres_token" "8" \
"--video_fps" "1" \
"--highres" "True" \
"--drop_threshold" "0.8" \
"--frame_pos" "False")

am_i_testing="0"
if [ "$am_i_testing" -eq 1 ]; then
    export CUDA_LAUNCH_BLOCKING=1 
    export TORCH_DISTRIBUTED_DEBUG=DETAIL 
    runtime="00:30:00"
    num_nodes=2
    sed -i "s|#SBATCH --partition=booster|#SBATCH --partition=develbooster|" ${copied_batch_file}
else
    unset CUDA_LAUNCH_BLOCKING
    unset TORCH_DISTRIBUTED_DEBUG
    runtime="04:00:00"
    num_nodes=8
fi


sed -i "s|#SBATCH --time=?|#SBATCH --time=${runtime}|" ${copied_batch_file}
sed -i "s|#SBATCH --nodes=?|#SBATCH --nodes=${num_nodes}|" ${copied_batch_file}
sed -i "s|#SBATCH --output=?|#SBATCH --output=${slurm_dir}/slurm_training_log.txt|" \
    ${copied_batch_file}
sed -i "s|#SBATCH --job-name=?|#SBATCH --job-name=${experiment_name}_${date_time}|" \
    ${copied_batch_file}

all_args=("${training_args[@]}" "${filesystem_related_args[@]}" "${data_args[@]}" \
"${llm_and_freezing_args[@]}" "${longvu_algorithm_args[@]}")
printf -v all_args_str "\"%s\" ${s} \\\n" "${all_args[@]}"
#echo -e $all_args_str
sed -i "s|all_args_set_by_submit=()|all_args_set_by_submit=($all_args_str)|" \
${copied_batch_file}
sbatch ${copied_batch_file} ${experiment_dir} $@
#bash ${copied_batch_file} ${experiment_dir} $@