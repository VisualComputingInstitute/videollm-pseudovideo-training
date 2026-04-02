
source scripts/julich/setup_env.sh
experiment_name=$1
experiment_dir=${experiments_dir}/${experiment_name}
slurm_dir=${experiment_dir}/slurm
lin2k_path=${PROJECT}/workspaces/longvu_workspace/val_data/lin2k

log_file=${slurm_dir}/slurm_eval_Linear2k.txt


sbatch --job-name="${experiment_name}_eval_Linear2k" --output=$log_file \
    --time=00:15:00 \
    scripts/julich/sbatch_tvbench_format.sh $experiment_dir Linear2k $lin2k_path