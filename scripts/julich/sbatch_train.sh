#!/bin/bash -x

#SBATCH --account=efficientvid

#SBATCH --partition=booster

#SBATCH --nodes=?

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=48

#SBATCH --job-name=?

#SBATCH --gres=gpu:4

#SBATCH --time=?

#SBATCH --array=0-10%1

#SBATCH --output=?

#SBATCH --open-mode=append

set -e

source scripts/julich/setup_env.sh
experiment_dir=$1
shift 1
cd $experiment_dir/code

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
#JUWELS peculiarity: allow communication over different InfiniBand cells.
master_addr="${master_addr}i.juwels"

deepspeed_cfg="scripts/zero2.json"

all_args_set_by_submit=()

if [ -f "${experiment_dir}/slurm/training_complete.flag" ]; then
    echo "training_complete.flag has been detected; canceling remaining SLURM tasks"
    scancel "$SLURM_ARRAY_JOB_ID"
    exit 0    
else
    srun --unbuffered \
        python -m torch.distributed.run --nproc_per_node=4 --nnodes=$SLURM_NNODES \
        --rdzv_id=29041997 --rdzv_backend=c10d --rdzv_endpoint=${master_addr} \
        longvu/train_modified.py "${all_args_set_by_submit[@]}" \
        --deepspeed $deepspeed_cfg $@
fi