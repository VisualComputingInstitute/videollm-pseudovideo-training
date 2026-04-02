#!/bin/bash -x

jutil env activate -p efficientvid
if ! ml 2>&1 | grep -qw "Stages/2024"; then
    module load Stages/2024
fi
module load PyTorch torchvision
source ${PROJECT}/venvs/longvu_juwels/bin/activate

export TRITON_CACHE_DIR="$(mktemp -d)"
export HF_HUB_OFFLINE=1
export HF_HOME="${PROJECT}/huggingface_cache"
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1

export longvu_workspace="${PROJECT}/workspaces/longvu_workspace"
export experiments_dir="${longvu_workspace}/experiments"

export known_tvbench_style_datasets=("tvbench" "one_obj_lin" "more_objs_lin" \
"four_new_colors" "more_objs_lin_count")
#"one_obj_lin_buggy" "four_new_colors")