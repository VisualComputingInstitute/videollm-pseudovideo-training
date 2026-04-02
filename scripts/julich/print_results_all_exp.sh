#!/bin/bash -x

set -e
source scripts/julich/setup_env.sh

pad=$(printf '%0.1s' "-"{1..200})
leftpadlength=50
rightpadlength=120

containing_dir_inside_base=${1}
shift 1
for dir in ${experiments_dir}/${containing_dir_inside_base}/*/; do
    if [ -d "$dir" ]; then
        dirname=${containing_dir_inside_base}/$(basename "$dir")

        print_str="BEGIN RESULTS FOR EXPERIMENT: ${dirname}"
        printf '%*.*s' 0 $leftpadlength "$pad"        
        printf "$print_str"
        printf '%*.*s' 0 $((rightpadlength - ${#print_str})) "$pad"
        printf '\n\n'

        bash ${PROJECT}/code_repos/longvu/scripts/julich/print_results.sh $dirname $@
        
        print_str="END RESULTS FOR EXPERIMENT: ${dirname}"
        printf '%*.*s' 0 $leftpadlength "$pad"
        printf "$print_str"
        printf '%*.*s' 0 $((rightpadlength - ${#print_str})) "$pad"
        printf '\n\n'  
    fi
done