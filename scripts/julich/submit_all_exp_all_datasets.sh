#!/bin/bash -x

set -e
source scripts/julich/setup_env.sh

containing_dir_inside_base=${1}
shift 1
for dir in ${experiments_dir}/${containing_dir_inside_base}/*/; do
    if [ -d "$dir" ]; then
        dirname=${containing_dir_inside_base}/$(basename "$dir")
        bash ${PROJECT}/code_repos/longvu/scripts/julich/submit_all_tvbench_format.sh $dirname $@
    fi
done