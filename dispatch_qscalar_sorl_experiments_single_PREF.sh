#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_qscalar_sorl_experiments_single_PREF.sh <environment> <gamma> <noise> <pref> <seed_file>
# ./dispatch_qscalar_sorl_experiments_single_PREF.sh rsense 0.997 0.7 0.5 seedlist.dat
mkdir -p logfiles

env=$1
gamma=$2
noise=$3
pref=$4
seed_filename=$5

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    ## reading each line
    log_filename=$1-qscalar-g$2-n$3-p$4-$line
    echo "$log_filename"
    python ./qscalar_sorl.py --env="$1" --gamma="$2" --noise="$3" --pref="$4" --seed="$line"  >> logfiles/"$log_filename" 2>&1 ;
    
done < "$seed_filename"

