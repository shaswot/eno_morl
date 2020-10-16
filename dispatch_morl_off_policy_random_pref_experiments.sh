#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_morl_off_policy_random_pref_experiments.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_morl_off_policy_random_pref_experiments.sh cenp 0.997 0.7 seedlist.dat
mkdir -p logfiles

env=$1
gamma=$2
noise=$3
seed_filename=$4

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    ## reading each line
    log_filename=$1-morl_off_policy_random_pref_diff_gamma_v2-g$2-n$3-$line
    echo "$log_filename"
    python ./morl_off_policy_random_pref_diff_gamma_v2.py --env="$1" --gamma="$2" --noise="$3" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

done < "$seed_filename"

