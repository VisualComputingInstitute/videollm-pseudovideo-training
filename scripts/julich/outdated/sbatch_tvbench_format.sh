#!/bin/bash -x

#SBATCH --account=efficientvid

#SBATCH --partition=booster

#SBATCH --cpus-per-task=48

#SBATCH --gres=gpu:4

#SBATCH --nodes=2
echo "hello"
set -e

source scripts/julich/setup_env.sh
experiment_dir=$1
dataset_name=$2
data_path=$3

#this tells apart llama from qwen and should have been created by the
#training run
version_file=${experiment_dir}/version_file.txt
version=$(<$version_file)

cd ${experiment_dir}/code
out_dir=${experiment_dir}/eval_results/${dataset_name}
mkdir -p ${out_dir}

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
#JUWELS peculiarity: allow communication over different InfiniBand cells.
master_addr="${master_addr}i.juwels"

#just to be sure since longvu seemed to have trouble placing processes 
#on GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
srun --unbuffered python -m torch.distributed.launch --nproc-per-node=4 \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=29041997 --rdzv_backend=c10d --rdzv_endpoint=${master_addr} \
    eval/eval_tvbench_format.py --data_path ${data_path} \
    --dataset_name $dataset_name \
    --model_path ${experiment_dir}/checkpoints \
    --all_res_output_file ${out_dir}/all_results.json \
    --acc_output_file ${out_dir}/acc_results.json \
    --version ${version}