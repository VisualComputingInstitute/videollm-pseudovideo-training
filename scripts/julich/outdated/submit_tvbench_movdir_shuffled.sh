
source scripts/julich/setup_env.sh
experiment_name=$1
experiment_dir=${experiments_dir}/${experiment_name}
slurm_dir=${experiment_dir}/slurm
tvbench_path=${PROJECT}/workspaces/videollava_workspace/eval/TVBench

log_file=${slurm_dir}/slurm_eval_TVBenchMovDirShuffled.txt

sbatch --job-name="${experiment_name}_eval_TVBenchMovDirShuffled" \
    --output=$log_file --time=00:15:00 \
    scripts/julich/sbatch_tvbench_format.sh $experiment_dir \
    TVBenchMoveDirShuffled $tvbench_path