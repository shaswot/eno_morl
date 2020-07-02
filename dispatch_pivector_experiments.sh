#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_pivector_experiments.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_pivector_experiments.sh cenp 0.997 0.35 seedlist.dat
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
    log_filename=$1-pivector_random_pref-g$2-n$3-$line
    echo "$log_filename"
#     python ./pivector.py --env="$1" --gamma="$2" --noise="$3" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &
    python ./pivector-random_pref.py --env="$1" --gamma="$2" --noise="$3" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &
#     python ./pivector-day_pref.py --env="$1" --gamma="$2" --noise="$3" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

done < "$seed_filename"

