#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_heuristics.sh <environment> <seed_file>
# ./dispatch_experiments.sh csense seedlist.dat

env=$1
seed_filename=$2

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    # Naive-heuristics
    log_filename=$1-naive-$line
    python ./naive_heuristics.py --env="$1" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

#     # K-heuristics
#     log_filename=$1-k-$line
#     python ./k_heuristics.py --env="$1" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &
    
    echo "$log_filename"

#    
# 2>&1: Redirect stderr to "where stdout is currently going". 
# In this case, that is a file opened in append mode. 
# In other words, the &1 reuses the file descriptor which stdout currently uses.
#
    
done < "$seed_filename"

