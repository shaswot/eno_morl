#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_pivector_yrpref_test.sh <seed_file>
# ./dispatch_pivector_experiments.sh seedlist.dat
mkdir -p logfiles

seed_filename=$1

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    ## reading each line
    log_filename=pivector_yr_pref-$line
    echo "$log_filename"
    python ./running_yrpref_pivector.py --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

done < "$seed_filename"

