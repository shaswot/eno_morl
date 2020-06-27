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

for pref in  0.6 0.7 #0.8 0.9 1.0 #0.0 0.1  0.2 0.3 0.4 0.5
do
    ./dispatch_c_morl_experiments_single_PREF.sh "$1" "$2" "$3" "$pref" "$4" &
done