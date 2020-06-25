#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_experiments.sh <environment> <seed_file>
# ./dispatch_experiments.sh rsense seedlist.dat

# get file with the list of seeds
experiment=$1
filename=$2
echo "Seedfile = $filename"
while IFS= read -r line
do
    ## reading each line
    echo "$line"
    python ./run_experiment.py --seed="$line" --gamma=0.997 --noise=0.3 "$1" & >> scriptlog.dat
    
done < "$filename"

