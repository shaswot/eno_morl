#!/bin/bash

## USAGE
## Run each experiment with seeds specified seedfile
# ./dispatch_c_morl_experiments_ALL_PREF.sh <environment> <gamma> <noise> <seed_file>
# ./dispatch_c_morl_experiments_ALL_PREF.sh cenp 0.997 0.7 seedlist.dat
mkdir -p logfiles

env=$1
gamma=$2
noise=$3
seed_filename=$4

for pref in  0.2 0.5 0.8 
do
    ./dispatch_c_morl_experiments_single_PREF.sh "$1" "$2" "$3" "$pref" "$4" &
done