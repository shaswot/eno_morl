#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_sorl_experiments.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_sorl_experiments.sh rsense 0.997 0.7 seedlist.dat

env=$1
gamma=$2
noise=$3
seed_filename=$4

# get file with the list of seeds
echo "Seedfile = $seed_filename"
while IFS= read -r line
do
    ## reading each line
    log_filename=$1-g$2-n$3-$line
    echo "$log_filename"
    python ./sorl.py --env="$1" --gamma="$2" --noise="$3" --seed="$line"  >> logfiles/"$log_filename" 2>&1 &

#    
# 2>&1: Redirect stderr to "where stdout is currently going". 
# In this case, that is a file opened in append mode. 
# In other words, the &1 reuses the file descriptor which stdout currently uses.
#
    
done < "$seed_filename"

