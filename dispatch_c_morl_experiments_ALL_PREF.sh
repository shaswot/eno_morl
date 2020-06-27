#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_c_morl_experiments_ALL_PREF.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_c_morl_experiments_ALL_PREF.sh cenp 0.997 0.7 seedlist.dat

env=$1
gamma=$2
noise=$3
seed_filename=$4

for pref in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    # get file with the list of seeds
    echo "Seedfile = $seed_filename"
    while IFS= read -r line
    do
        ## reading each line
        log_filename=$1sense-g$2-n$3-p$pref-$line
        echo "$log_filename"
        python ./c_morl.py --env="$1" --gamma="$2" --noise="$3" --pref="$pref" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &   
    done < "$seed_filename"
done