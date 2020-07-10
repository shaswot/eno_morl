#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_morl_off_policy_random_pref_experiments.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_morl_off_policy_random_pref_experiments.sh cenp 0.997 0.7 seedlist.dat
mkdir -p logfiles

seed_filename=$1

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    ## reading each line
    log_filename=task3-$line
    echo "$log_filename"
    python ./task3.py --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

done < "$seed_filename"

